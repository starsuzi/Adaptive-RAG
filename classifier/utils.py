import torch
import transformers
from transformers import (
    CONFIG_MAPPING,
    MODEL_MAPPING,
    AutoConfig,
    AutoModelForSeq2SeqLM,
    AutoTokenizer,
    DataCollatorForSeq2Seq,
    SchedulerType,
    get_scheduler,
)
import datasets
import numpy as np
import math

from transformers.trainer_utils import EvalLoopOutput, EvalPrediction, get_last_checkpoint


def load_model(args):
    # Load pretrained model and tokenizer
    # In distributed training, the .from_pretrained methods guarantee that only one local process can concurrently
    # download model & vocab.
    if args.config_name:
        config = AutoConfig.from_pretrained(args.config_name)
    elif args.model_name_or_path:
        config = AutoConfig.from_pretrained(args.model_name_or_path)
    else:
        config = CONFIG_MAPPING[args.model_type]()
        logger.warning("You are instantiating a new config instance from scratch.")

    if args.tokenizer_name:
        tokenizer = AutoTokenizer.from_pretrained(args.tokenizer_name, use_fast=not args.use_slow_tokenizer)
    elif args.model_name_or_path:
        tokenizer = AutoTokenizer.from_pretrained(args.model_name_or_path, use_fast=not args.use_slow_tokenizer)
    else:
        raise ValueError(
            "You are instantiating a new tokenizer from scratch. This is not supported by this script."
            "You can do it from another script, save it, and load it from here, using --tokenizer_name."
        )

    if args.model_name_or_path:
        model = AutoModelForSeq2SeqLM.from_pretrained(
            args.model_name_or_path,
            from_tf=bool(".ckpt" in args.model_name_or_path),
            config=config,
        )
    else:
        logger.info("Training new model from scratch")
        model = AutoModelForSeq2SeqLM.from_config(config)
        
    return model, tokenizer


def preprocess_dataset(args, raw_datasets):
    # Preprocessing the datasets.
    # First we tokenize all the texts.
    if args.do_eval:
        column_names = raw_datasets[args.val_column].column_names
    else :
        column_names = raw_datasets[args.train_column].column_names
    
    # Get the column names for input/target.
    question_column = args.question_column
    if question_column not in column_names:
        raise ValueError(
            f"--question_column' value '{args.question_column}' needs to be one of: {', '.join(column_names)}"
        )

    answer_column = args.answer_column
    if answer_column not in column_names:
        raise ValueError(
            f"--answer_column' value '{args.answer_column}' needs to be one of: {', '.join(column_names)}"
        )

    return question_column, answer_column


def preprocess_features_function(examples, args, raw_datasets, tokenizer):
    question_column, answer_column = preprocess_dataset(args, raw_datasets)

    # Temporarily set max_answer_length for training.
    max_answer_length = args.max_answer_length
    padding = "max_length" if args.pad_to_max_length else False
    max_seq_length = min(args.max_seq_length, tokenizer.model_max_length)



    # Some of the questions have lots of whitespace on the left, which is not useful and will make the
    # truncation of the context fail (the tokenized question will take a lots of space). So we remove that
    # left whitespace

    examples[question_column] = ['{}'.format(q.strip()) for q in examples[question_column]]


    # Tokenize our examples with truncation and maybe padding, but keep the overflows using a stride. This results
    # in one example possible giving several features when a context is long, each of those features having a
    # context that overlaps a bit the context of the previous feature.
    model_inputs = tokenizer(
        examples[question_column],
        truncation=True,
        max_length=max_seq_length,
        stride=args.doc_stride,
        return_overflowing_tokens=True,
        return_offsets_mapping=True,
        padding=padding,
    )   

    targets = examples[answer_column]

    # Tokenize targets with the `text_target` keyword argument
    labels = tokenizer(text_target=targets, max_length=max_answer_length, padding=padding, truncation=True)

    # If we are padding here, replace all tokenizer.pad_token_id in the labels by -100 when we want to ignore
    # padding in the loss.
    if padding == "max_length" and args.ignore_pad_token_for_loss:
        labels["input_ids"] = [
            [(l if l != tokenizer.pad_token_id else -100) for l in label] for label in labels["input_ids"]
        ]

    # Since one example might give us several features if it has a long context, we need a map from a feature to
    # its corresponding example. This key gives us just that.
    sample_mapping = model_inputs.pop("overflow_to_sample_mapping")

    # For evaluation, we will need to convert our predictions to substrings of the context, so we keep the
    # corresponding example_id and we will store the offset mappings.
    model_inputs["example_id"] = []
    # Augment the overflowing tokens to the labels
    labels_out = []

    for i in range(len(model_inputs["input_ids"])):
        # One example can give several spans, this is the index of the example containing this span of text.
        sample_index = sample_mapping[i]
        model_inputs["example_id"].append(examples["id"][sample_index])
        labels_out.append(labels["input_ids"][sample_index])

    model_inputs["labels"] = labels_out
    return model_inputs


# Post-processing:
def post_processing_function(
    tokenizer, args, raw_datasets, examples: datasets.Dataset, features: datasets.Dataset, outputs, stage="eval"
):
    # Decode the predicted tokens.
    preds = outputs
    if isinstance(preds, tuple):
        preds = preds[0]
    # Replace -100s used for padding as we can't decode them
    preds = np.where(preds != -100, preds, tokenizer.pad_token_id)
    
    decoded_preds = tokenizer.batch_decode(preds, skip_special_tokens=True)
    return decoded_preds 



# Create and fill numpy array of size len_of_validation_data * max_length_of_output_tensor
def create_and_fill_np_array(all_gen_tokens, dataset, max_len):
    """
    Create and fill numpy array of size len_of_validation_data * max_length_of_output_tensor
    Args:
        all_gen_tokens(:obj:`tensor`):
            This is the output predictions of the model.
        eval_dataset: Evaluation dataset
        max_len(:obj:`int`):
            The maximum length of the output tensor. ( See the model.eval() part for more details )
    """
    
    step = 0
    # create a numpy array and fill it with -100.
    gen_toks_concat = np.full((len(dataset), max_len), -100)
    # Now since we have create an array now we will populate it with the outputs gathered using accelerator.gather_for_metrics
    for i, gen_tok in enumerate(all_gen_tokens):  # populate columns
        # We have to fill it such that we have to take the whole tensor and replace it on the newly created array
        # And after every iteration we have to change the step
        #import pdb; pdb.set_trace()
        batch_size = gen_tok.shape[0]
        cols = gen_tok.shape[1]

        if step + batch_size < len(dataset):
            gen_toks_concat[step : step + batch_size, :cols] = gen_tok
        else:
            gen_toks_concat[step:, :cols] = gen_tok[: len(dataset) - step]

        step += batch_size

    return gen_toks_concat


def prepare_scheduler(args, accelerator, dataloader, optimizer, max_train_steps, train_epoch):
    overrode_max_train_steps = False

    num_update_steps_per_epoch = math.ceil(len(dataloader) / args.gradient_accumulation_steps)

    if max_train_steps is None:
        max_train_steps = train_epoch * num_update_steps_per_epoch
        overrode_max_train_steps = True

    lr_scheduler = get_scheduler(
        name=args.lr_scheduler_type,
        optimizer=optimizer,
        num_warmup_steps=args.num_warmup_steps * args.gradient_accumulation_steps,
        num_training_steps=max_train_steps * args.gradient_accumulation_steps,
    )

    lr_scheduler = accelerator.prepare(lr_scheduler)

    num_update_steps_per_epoch = math.ceil(len(dataloader) / args.gradient_accumulation_steps)

    # We need to recalculate our total training steps as the size of the training dataloader may have changed.
    if overrode_max_train_steps:
        max_train_steps = train_epoch * num_update_steps_per_epoch
    # Afterwards we recalculate our number of training epochs
    train_epoch = math.ceil(max_train_steps / num_update_steps_per_epoch)

    return max_train_steps, train_epoch, lr_scheduler


def get_gold_answers(example):
    """helper function that retrieves all possible true answers from a squad2.0 example"""
    
    gold_answers = [answer["text"] for answer in example.answers if answer["text"]]

    # if gold_answers doesn't exist it's because this is a negative example - 
    # the only correct answer is an empty string
    if not gold_answers:
        gold_answers = [""]
        
    return gold_answers

def calculate_accuracy(gold_answers, predictions):
    total_acc_score = 0
    for (gold_answer, prediction) in zip(gold_answers, predictions):
        acc_score = int(gold_answer == prediction)
        total_acc_score = total_acc_score + acc_score

    final_acc_score = (total_acc_score / len(gold_answers)) * 100
    return final_acc_score

def calculate_accuracy_perClass(gold_answers, predictions):
    a_total_acc_score = 0
    b_total_acc_score = 0
    c_total_acc_score = 0

    a_gold_num = len([i for i in gold_answers if i == 'A'])
    b_gold_num = len([i for i in gold_answers if i == 'B'])
    c_gold_num = len([i for i in gold_answers if i == 'C'])

    a_pred_num = len([i for i in predictions if i == 'A'])
    b_pred_num = len([i for i in predictions if i == 'B'])
    c_pred_num = len([i for i in predictions if i == 'C'])

    for (gold_answer, prediction) in zip(gold_answers, predictions):
        # a
        a_acc_score = int(gold_answer == prediction == 'A')
        a_total_acc_score = a_total_acc_score + a_acc_score
        # b
        b_acc_score = int(gold_answer == prediction == 'B')
        b_total_acc_score = b_total_acc_score + b_acc_score
        # c
        c_acc_score = int(gold_answer == prediction == 'C')
        c_total_acc_score = c_total_acc_score + c_acc_score


    a_final_acc_score = (a_total_acc_score / a_gold_num) * 100 if a_gold_num != 0 else -1
    b_final_acc_score = (b_total_acc_score / b_gold_num) * 100 if b_gold_num != 0 else -1
    c_final_acc_score = (c_total_acc_score / c_gold_num) * 100 if c_gold_num != 0 else -1

    dict_final =  {'A (zero) acc' : a_final_acc_score, 'B (single) acc' : b_final_acc_score, 'C (multi) acc' : c_final_acc_score,
    'A (zero) pred num' : a_pred_num, 'B (single) pred num' : b_pred_num, 'C (multi) pred num' : c_pred_num,
    'A (zero) gold num' : a_gold_num, 'B (single) gold num' : b_gold_num, 'C (multi) gold num' : c_gold_num}
    return dict_final
