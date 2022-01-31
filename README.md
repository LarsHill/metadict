<div align="center">
<img src="logo/fluid_ml_logo.png" width="400px">

_Develop ML pipelines fluently with no boilerplate code. Focus only on your tasks and not the boilerplate!_

---

<p align="center">
  <a href="#key-features">Key Features</a> •
  <a href="#getting-started">Getting Started</a> •
  <a href="#examples">Examples</a> •
  <a href="#citation">Citation</a>
</p>

[![Python 3.6](https://img.shields.io/badge/python-3.6+-blue.svg)](https://www.python.org/downloads/release/python-360/)
[![License](https://img.shields.io/badge/License-Apache%202.0-blue.svg)](https://opensource.org/licenses/Apache-2.0)
[![CircleCI](https://circleci.com/gh/fluidml/fluidml/tree/main.svg?style=shield)](https://circleci.com/gh/fluidml/fluidml/tree/main)
[![codecov](https://codecov.io/gh/fluidml/fluidml/branch/main/graph/badge.svg?token=XG4UDWF8RE)](https://codecov.io/gh/fluidml/fluidml)
[![Contributor Covenant](https://img.shields.io/badge/Contributor%20Covenant-v2.0%20adopted-ff69b4.svg)](https://github.com/fluidml/fluidml/blob/main/CODE_OF_CONDUCT.md)

</div>

---

**FluidML** is a lightweight framework for developing machine learning pipelines.

<div align="center">
<img src="logo/fluidml_example.gif" width="70%" />
</div>

Developing machine learning models is a challenging process, with a wide range of sub-tasks: data collection, pre-processing, model development, hyper-parameter tuning and deployment. Each of these tasks is iterative in nature and requires lot of iterations to get it right with good performance.

Due to this, each task is generally developed sequentially, with artifacts from one task being fed as inputs to the subsequent tasks. For instance, raw datasets are first cleaned, pre-processed, featurized and stored as iterable datasets (on disk), which are then used for model training. However, this type of development can become messy and un-maintenable quickly for several reasons:

- pipeline code may be split across multiple scripts whose dependencies are not modeled explicitly
- each of this task contains boilerplate code to collect results from previous tasks (eg: reading from disk)
- hard to keep track of task artifacts and their different versions
- hyper-parameter tuning adds further complexity and boilerplate code

## Key Features

FluidML provides following functionalities out-of-the-box:

- **Task Graphs** - Create ML pipelines or task graph using simple APIs
- **Results Forwarding** - Results from tasks are automatically forwarded to downstream tasks based on dependencies
- **Parallel Processing** - Execute the task graph parallely with multi-processing
- **Grid Search** - Extend the task graph by enabling grid search on tasks with just one line of code
- **Result Caching** - Task results are cached in a results store (eg: Local File Store or a MongoDB Store) and made available for subsequent runs without executing the tasks again and again
- **Flexibility** - Provides full control on your task implementations. You are free to choose any framework of your choice (Sklearn, TensorFlow, Pytorch, Keras, or any of your favorite library)

---

## Getting Started

### Installation

#### 1. From Pip
Simply execute:  
```bash
$ pip install fluidml
```

#### 2. From Source
1. Clone the repository,
2. Navigate into the cloned directory (contains the setup.py file),
3. Execute `$ pip install .`

**Note:** To run demo examples, execute `$ pip install fluidml[examples]` (Pip) or `$ pip install .[examples]` (Source) to install the additional requirements.

### Minimal Example

This minimal toy example showcases how to get started with FluidML.
For real machine learning examples, check the "Examples" section below.

#### 1. Basic imports

First, we import necessary classes from FluidML.

```Python
from fluidml import Task, Flow, Swarm
from fluidml.common import Resource
from fluidml.flow import GridTaskSpec, TaskSpec
from fluidml.storage import MongoDBStore, LocalFileStore, ResultsStore
from fluidml.visualization import visualize_graph_in_console
```

#### 2. Define Tasks

Next, we define some toy machine learning tasks. A Task can be implemented as a function or as a class inheriting from our `Task` class.

In case of the class approach, each task should implement the `run()` method, which takes some inputs and performs the desired functionality. These inputs are actually the results from predecessor tasks and are automatically forwarded by FluidML based on registered task dependencies. If the task has any hyper-parameters, they can be defined as arguments in the constructor. Additionally, within each task, users have access to methods and attributes like `self.save()` and `self.resource` to save its result and access task resources (more on that later).

```Python
class MyTask(Task):
    def __init__(self, kwarg_1, kwarg_2):
        ...
    def run(self, result_1, result2):
        ...
```

or

```Python
def my_task(result_1, result_2, kwarg_1, kwarg_2, task: Task):
    ...
```

In the case of defining the task as callable, an extra task object is provided to the task,
which makes important internal attributes and functions like `task.save()` and `task.resource` available to the user.

Below, we define standard machine learning tasks such as dataset preparation, pre-processing, featurization and model training using Task classes.
Notice that:

- Each task is implemented individually and it's clear what the inputs are (check arguments of `run()` method)
- Each task saves its results using `self.save(...)` by providing the object to be saved and a unique name for it. This unique name corresponds to input names in successor task definitions.

```Python
class DatasetFetchTask(Task):
    def run(self):
        ...
        # For InMemoryStore (default) and MongoDBStore type_ is NOT required
        # For LocalFileStore type_ IS required               
        self.save(obj=data_fetch_result, name='data_fetch_result', type_='json')


class PreProcessTask(Task):
    def __init__(self, pre_processing_steps: List[str]):
        super().__init__()
        self._pre_processing_steps = pre_processing_steps

    def run(self, data_fetch_result):
        ...
        self.save(obj=pre_process_result, name='pre_process_result')


class TFIDFFeaturizeTask(Task):
    def __init__(self, min_df: int, max_features: int):
        super().__init__()
        self._min_df = min_df
        self._max_features = max_features

    def run(self, pre_process_result):
        ...
        self.save(obj=tfidf_featurize_result, name='tfidf_featurize_result')


class GloveFeaturizeTask(Task):
    def run(self, pre_process_result):
        ...
        self.save(obj=glove_featurize_result, name='glove_featurize_result')


class TrainTask(Task):
    def __init__(self, max_iter: int, balanced: str):
        super().__init__()
        self._max_iter = max_iter
        self._class_weight = "balanced" if balanced else None

    def run(self, tfidf_featurize_result, glove_featurize_result):
        ...
        self.save(obj=train_result, name='train_result')


class EvaluateTask(Task):
    def run(self, train_result):
        ...
        self.save(obj=evaluate_result, name='evaluate_result')
```

#### 3. Task Specifications

Next, we can create the defined tasks with their specifications. 
We now only write their specifications, later these are used to create real instances of tasks by FluidML.
For each Task specification, we also add a list of result names that the corresponding task _publishes_. 
Each published result object will be considered when results are automatically collected for a successor task.

Note: The `config` argument holds the configuration of the task (ie. hyper-parameters). 
It has to be a dictionary (possibly nested), which is `json` serializable. 
That means a `TypeError` is thrown if the dictionary contains objects, e.g. an `np.array`, that cannot be serialized by Python's json encoder.

Additionally, if there are other parameters to the task, including objects of non-standard types (eg. models, tensors, files, etc), they can be passed using `additional_kwargs` argument.


```Python
dataset_fetch_task = TaskSpec(task=DatasetFetchTask, publishes=['data_fetch_result'])
pre_process_task = TaskSpec(task=PreProcessTask,
                            config={
                                "pre_processing_steps": ["lower_case", "remove_punct"]},
                            publishes=['pre_process_result'])
featurize_task_1 = TaskSpec(task=GloveFeaturizeTask,
                            publishes=['glove_featurize_result'])
featurize_task_2 = TaskSpec(task=TFIDFFeaturizeTask, config={"min_df": 5, "max_features": 1000},
                            publishes=['tfidf_featurize_result'])
train_task = TaskSpec(task=TrainTask, config={"max_iter": 50, "balanced": True},
                      publishes=['train_result'])
evaluate_task = TaskSpec(task=EvaluateTask, publishes=['evaluate_result'])
```

#### 4. Registering task dependencies

Here we create the task graph by registering dependencies between the tasks. In particular, for each task specifier, you can register a list of predecessor tasks using the `requires()` method.

```Python
pre_process_task.requires(dataset_fetch_task)
featurize_task_1.requires(pre_process_task)
featurize_task_2.requires(pre_process_task)
train_task.requires([dataset_fetch_task, featurize_task_1, featurize_task_2])
evaluate_task.requires([dataset_fetch_task, featurize_task_1, featurize_task_2, train_task])
```

#### 5. [optional] Define and instantiate Resources to share across all Tasks

Additionally, you can pass a list of resources (eg. GPU devices) that are made available to the workers, which forward them to the corresponding tasks.
You just have to create your own Resource dataclass, which inherits from our `Resource` interface. In this dataclass you can define all resources, e.g. the cuda device, which automatically is made available to all tasks through the `self.resource` or `task.resource` attribute.

```python
@dataclass
class TaskResource(Resource):
    device: str
```

Let's assume our resources consist of a list of cuda device ids, e.g. `['cuda:0', 'cuda:1', 'cuda:0', 'cuda:1']`, and we set `num_workers=4`.
Then we can create our list of resources object with a simple list comprehension:

```python
# create list of resources
resources = [TaskResource(device=devices[i]) for i in range(num_workers)]
```

#### 6. [optional] Results Store/Caching

By default, results of tasks are stored in an `InMemoryStore`, which might be impractical for large datasets/models. 
Also, the results are not persistent. To have persistent storage, FluidML provides two fully implemented `ResultsStore` namely `LocalFileStore` and `MongoDBStore`.

Additionally, users can provide their own results store to `Swarm` by inheriting from `ResultsStore` class and implementing `load()`, `save()` and `delete()`. 
Note these methods rely on task name and its config parameters, which act as lookup-key for results. 
In this way, tasks are skipped by FluidML when task results are already available for the given config. 
But users can override and force execute tasks by passing `force` parameter to the `Flow`.

```Python
class MyResultsStore(ResultsStore):
    def load(self, name: str, task_name: str, task_unique_config: Dict, **kwargs) -> Optional[Any]:
        """ Query method to load an object based on its name, task_name and task_config if it exists """
        raise NotImplementedError

    def save(self, obj: Any, name: str, type_: str, task_name: str, task_unique_config: Dict, **kwargs):
        """ Method to save/update any artifact """
        raise NotImplementedError

    def delete(self, name: str, task_name: str, task_unique_config: Dict):
        """ Method to delete any artifact """
        raise NotImplementedError
```

We can instantiate for example a `LocalFileStore`

```python
results_store = LocalFileStore(base_dir='/some/dir')
```

and pass it in the next step to `Swarm` to enable persistent results storing.

#### 7. [optional] Configure Logging

FluidML internally utilizes Python's `logging` library. However, we refrain from configuring a logger object with handlers
and formatters since each user has different logging needs and preferences. Hence, if you want to use FluidML's logging
capability, you just have to do the configuration yourself. For convenience, we provide a simple utility function which
configures a visually appealing logger (using a specific handler from the [rich](https://github.com/willmcgugan/rich) library).

```python
from fluidml.common.logging import configure_logging
configure_logging()
```

**Note**: If you want to use logging in your application (e.g. within FluidML Tasks) but want to disable all FluidML internal logging messages you can
simply call

```python
logging.getLogger('fluidml').propagate = False
```

#### 8. Run tasks using Flow and Swarm

Now that we have all the tasks specified, we can just run the task graph. 
For that, we have to create an instance of the`Swarm` class, by specifying a number of workers (`n_dolphins` :wink:).
If `n_dolphin` is not set, it defaults internally to the number of CPU's available to the machine.
Additionally, we can provide one of our persistent result stores (defaults to `InMemoryStore` if no store is provided).

Next, we create an instance of the `Flow` class which will construct the task graph via the `create()` method. 
Finally, we run the pipeline by calling `flow.run()`. Internally, `Swarm` executes the graph in parallel while considering the registered dependencies.

```Python
tasks = [dataset_fetch_task, pre_process_task, featurize_task_1,
         featurize_task_2, train_task, evaluate_task]

with Swarm(n_dolphins=2,                        # optional (defaults to number of CPU's)
           resources=resources,                 # optional
           return_results=True,                 # optional
           results_store=results_store,         # optional
           ) as swarm:
    flow = Flow(swarm=swarm)
    flow.create(task_specs=tasks)
    
    # optional graph visualization
    visualize_graph_in_console(graph=flow.task_spec_graph,
                               use_pager=True,    # optional (defaults to True)
                               use_unicode=False  # optional (defaults to False)
                               )
    results = flow.run(force=None)
```
**Note 1**: After calling `flow.create()` we have access to the created task specifier graph via `flow.task_spec_graph`. 
This constructed graph can be rendered on the console using fluidML's visualization routine `visualize_graph_in_console(graph=flow.task_spec_graph)`. 
The default arguments `use_pager=True` and `use_unicode=False` will render the graph in ascii within a pager for horizontal scralling support. 
If `use_pager=False` the graph is simply printed and if `use_unicode=True` a visually more appealing unicode character set is used for console rendering. 
Note not every console supports unicode characters.


See below the visualization of the task specifier graph from our example:

<div align="center">
<img src="logo/task_spec_graph.png" width="500px">
</div>

**Note 2**: We can provide a `force='all'` to the `run` method in order to force execute all tasks in the pipeline regardless of previously saved results.
Alternatively, `force` can take a concrete task name or a list of task names as argument. 
If all successor tasks of a provided task name should also be force execute you simply write `force='PreProcessTask+'`. 
The "+" registers all successor tasks for force execution as well.

**Note 3**: If the `InMemoryStore` is used, results of all the tasks are always returned by `flow.run()`, so that the user can store them manually. 
For the other shipped storages the user has the option to return or not return results (`return_results=True/False`). 
Task results can be accessed via task names, e.g. `results["EvaluationTask"]`. 
Our shipped result stores can be utilized to fetch specific task results from the returned result dictionary at any point via `results_store.load()`.

### Grid Search

Users can easily enable grid search for their tasks with just one line of code. 
To enable grid search on a particular task, we just have to wrap it with `GridTaskSpec` instead of `TaskSpec` and specify hyper-parameters in the grid search config argument `gs_config` as follows:

```Python
train_task = GridTaskSpec(task=TrainTask,
                          gs_config={"max_iter": [50, 100],
                                     "balanced": [True, False],
                                     "layers": [[50, 100, 50]]},
                          gs_expansion_method='product'  # or 'zip'
                          )
```

That's it! Internally, Flow expands this task into 4 tasks with provided cross product combinations of `max_iter` and `balanced`. 
Alternatively, one can select `zip` as the expansion method, which would result in 2 expanded tasks, with the respective `max_iter` and `balanced` combinations of `(50, True), (100, False)`. 
Generally, all values of type `List` will be unpacked to form grid search combinations. 

Additional task parameters can be provided via `additional_kwargs` argument. However, note that these parameters are not subject to grid search expansion. 

The expanded task graph object is available via `flow.task_graph` after calling `flow.create()`. 
As before, we can again visualize the expanded task graph in the console via `visualize_graph_in_console(graph=flow.task_graph)`.

<div align="center">
<img src="logo/task_graph.png">
</div>

Note: Since we use Python's `json` module to serialize the provided configs, we don't distinguish between types `List` and `Tuple`. 
In fact, tuples are internally converted to lists, and thus get expanded in the same way as described above.

If a list itself is an argument and should not be expanded, it has to be wrapped again in a list. 
That is why `layers` is not considered for different grid search realizations. 
Further, any successor tasks (for instance, evaluate task) in the task graph will also be automatically expanded. 
Therefore, in our example, we would have 4 evaluate tasks, each one corresponding to the 4 train tasks.

Running a complete machine learning pipeline usually yields trained models for many grid search parameter combinations.
A common goal is then to automatically determine the best hyper-parameter setup and the best performing model.
FluidML enables just that by providing a `reduce=True` argument to the `TaskSpec` class. Hence, to automatically 
compare the 4 evaluate tasks and select the best performing model, we implement an additional `ModelSelectionTask`
which gets wrapped by our `TaskSpec` class.

```Python
class ModelSelectionTask(Task):
    def run(self, reduced_results: List[Dict[str, Dict]]):
        # from all trained models/hyper-parameter combinations, determine the best performing model
        ...

model_selection_task = TaskSpec(task=ModelSelectionTask, reduce=True, expects=['evaluate_result'])

model_selection_task.requires(evaluate_task)
```

The important `reduce=True` argument enables that a single `ModelSelectionTask` instance gets the reduced results
from all grid search expanded predecessor tasks. 

**Note**: When specifying a `reduce` task one can explicitly provide the expected inputs as list of strings to the `expects` argument of `TaskSpec`.
The expected inputs will be retrieved from the predecessor tasks and packed in a special `reduced_results` argument (the only input to the task's `run` method).
If `expects` is not provided all published predecessor task results will be retrieved and packed in `reduced_results`.
It is a list of dictionaries where each dictionary holds the results and config of one specific grid search parameter combination. For example:

```Python
reduced_results = [
    {'result': {'result_name_1': result_1,
                'result_name_2': result_2},
     'config': {...}  # first unique parameter combination config
     },
    {'result': {'result_name_1': result_1,
                'result_name_2': result_2},
     'config': {...}  # second unique parameter combination config
     }
]
```

---

## Examples

For real machine learning pipelines including grid search implemented with FluidML, check our
Jupyter Notebook tutorials:

- [Transformer based Sequence to Sequence Translation (PyTorch)](https://github.com/fluidml/fluidml/blob/main/examples/pytorch_transformer_seq2seq_translation/transformer_seq2seq_translation.ipynb)
- [Multi-class Text Classification (Sklearn)](https://github.com/fluidml/fluidml/blob/main/examples/sklearn_text_classification/sklearn_text_classification.ipynb)

---

## Citation

```
@article{fluid_ml,
  title = {FluidML - a lightweight framework for developing machine learning pipelines},
  author = {Ramamurthy, Rajkumar and Hillebrand, Lars},
  year = {2020},
  publisher = {GitHub},
  journal = {GitHub repository},
  howpublished = {\url{https://github.com/fluidml/fluidml}},
}
```
