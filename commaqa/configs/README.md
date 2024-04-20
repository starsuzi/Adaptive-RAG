# Dataset Configs

To create your own dataset with new KB, Agents and Theories, you would need to specify your own
dataset config file. This README describes the format of such a config file and the individual
objects within the config file. Refer to the [README](../dataset/README.md) for details about
building the dataset.

### Defining your own config
To define a new dataset configuration, create a JSONNET file with the following format:
```jsonnet
{
  version: 3.0,
  entities: Entities,
  predicates: Predicates,
  predicate_language: Predicate_Language,
  theories: Theories,
}
```
Each object in this config is described below. You may choose to format the jsonnet file using imported libsonnet files, e.g., in
[CommaQA-E](../../configs/commaqav1/explicit).

### Config Objects

#### Entities
A dictionary of `EntityType` to a list of strings that correspond to entities of this type.
E.g.
```jsonnet
{
    entities: {
        movie: [
            "Godfather",
            "Kill Bill",
        ],
        actor: [
            "Marlon Brando",
            "Uma Thurman",
        ],
        year: [
            "1990",
            "1995",
        ]
    }
}
```


#### Predicates
Each predicate defines a relation between `EntityType`s in the list of `entities`.
E.g.
```jsonnet
{
    acted_in: {
        args: ["movie", "actor"],
        nary: ["n", "1"],
        language: ["movie: $1 ; actor: $2", "$2 acted in $1."],
    }
}
```
Here the `args` specify the arguments of the `acted_in` predicate. `nary` specifies that the
relation is 1-to-many. Specifically, when this KB is grounded (via sampling), each movie can appear
multiple times in the KB relations but the actor can only appear once. This will essentially cause
each movie to have multiple actors and each actor to appear in only one movie. Changing `nary` to
`["n", "n"]` would make it a many-to-many relation. Note that it is also specify a tree (e.g. Isa)
or chain structure (e.g. successor) to your KB by setting `type` to `tree` or `chain` instead of
`nary` field. The `language` field specifies how the relation will be verbalized in text. This is not relevant for
the agents but defines the input context for black-box models. One of the verbalizations from the
`language` is randomly sampled for each KB fact when generating the context.


The `predicates` in the configuration file is a dictionary of `PredicateName` to the properties of
the predicate as described above. E.g.
```jsonnet
{
    predicates: {
     acted_in: {
            args: ["movie", "actor"],
            nary: ["n", "1"],
            language: ["movie: $1 ; actor: $2", "$2 acted in $1."],
     },
     released_in: {
            args: ["movie", "year"],
            nary: ["1", "n"],
            language: ["movie: $1 ; year: $2"],
     }
    }
}
```

#### Predicate Language
This field defines the questions that can be answered by each agent and how each agent answers these
questions given their KB. The configuration specifies a helper predicate, what `EntityType`s to use
for the question, the agent name, how the questions should be phrased and how the questions are
answered using the KB. For example,
```jsonnet
{
    "acted_a($1, ?)": {
      "init": {
        "$1": "movie"
      },
      "model": "text",
      "questions": [
        "Who all acted in the movie $1?",
        "Who are the actors in the movie $1?"
      ],
      "steps": [
        {
          "answer": "#1",
          "operation": "select",
          "question": "text_actor($1, ?)"
        }
      ]
    }
}
```
In this example:
  * `acted_a($1, ?)`: This is a helper predicate that will be used to define the theories later.
This question takes one input argument `$1` and returns the second argument `?` as the answer.
  * `init`: This dictionary specifies the entity type for each argument in the predicate name. In
this case the question takes one argument which is a movie.
  * `model`: This field specifies the agent name -- `text` (i.e. TextQA agent) in this case
  * `questions`: This list specifies the different formulations of this question that can be
answered by the agent. Here the `text` agent can answer questions about the actors of a movie using
either of these forms. Note that our synthetic dataset uses symbolic agents that need one of these
formulations to be used exactly.
  * `steps`: This field can be used to describe a multi-step procedure to answer these questions. In
this case, we defined a single-step procedure where the agent will take the input movie `$1` and
lookup the `text_actor` relation in the KB. The first argument of this KB lookup will be grounded to
the movie (`$1`) and the second argument (`?`) will be returned as the answer. The answer is named
as `#1` (this name can be used in future steps to refer to this answer). Refer to the paper for an
explanation about the `select` operation (and other possible operations).


#### Theories
Finally we define the theories that will be used to create the questions for the complex tasks based
on the components described above. Consider the following sample theory:
```jsonnet
{
      "init": {
        "$1": "nation"
      },
      "questions": [
        "What movies have people from the country $1 acted in?"
      ],
      "steps": [
        {
          "answer": "#1",
          "operation": "select",
          "question": "nation_p($1, ?)"
        },
        {
          "answer": "#2",
          "operation": "project_values_flat_unique",
          "question": "acted_m(#1, ?)"
        }
      ]
}
```
 * `init`: Again refers to the `EntityType` used to create the question
 * `questions`: List of possible formalizations of this question
 * `steps`: Multi-step procedure to execute this question. Refer to the explanation of
`steps` [above](#predicate-language). For explanation about `operation`, refer to our paper. The
`question` field refers to the helper predicate names introduced in the
[Predicate Language](#predicate-language) section