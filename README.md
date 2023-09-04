# Hosting Pytorchserving Models

This document describes the process around **creating**, **deploying**, and **managing** Pytorch-based and non-Pytorch-based models with
[Pytorchserving](https://github.com/pytorch/serve/blob/master/README.md). This is followed by the instructions specific to each Pytorchserving based model in this repository.


## General Concepts

### 1. The `.mar` file

Each model is bundled into a `.mar` (model archive) file that contains a
handler (`handler.py`), a model checkpoint (`checkpoint.pth`), additional files (packaged as `.zip`, `.tar`, `.tar.gz` etc.), requirements file (`requirements.txt`) etc and the models code.

A `.mar` file is like a regular zip file (**tip**: Run `unzip model.mar` and see the contents of the unzipped folder).

To build the `.mar` file we use the torch-model-archiver tool. Basic usage of this is as follows:
```
torch-model-archiver --model-name <model-name> --version <model_version_number> --handler model_handler
[:<entry_point_function_name>] [--model-file <path_to_model_architecture_file>] --serialized-file
<path_to_state_dict_file> [--extra-files <comma_seperarted_additional_files>] [--export-path <output-dir>
--model-path <model_dir>] [--runtime python3]
```
---

### 2. The life of a Pytorchserving model

When the model is loaded into the memory by pytorchserving, it first creates a temporary directory (e.g. `/tmp/f3bab7bfaab7bfce`) and copies the
contents of the `.mar` file into the temporary directory.

Pytorchserving adds this directory to `PYTHONPATH` (therefore, statements like `from handler import handle` will work).

Next, Pytorchserving installs the requirements from the given `requirements.txt` file. Note that the requirements file is optional, and a separate
environment is created for each model with the given requirements. The temporary directory serves as the install location of the packages. In this manner, additional dependencies can be shipped along with the `.mar` file.

Once the requirements are installed, pytorchserving attempts to load model by using the source code from the `handler` file. Either by using a handler class or by a handler function.

---

### 3. Creating custom handlers

Please refer to the official pytorch [documentation](https://pytorch.org/serve/custom_service.html) for information on how to build a custom handler (either class based or function based) and package it into a `.mar` file.


## 2. Using model-server helpers to build a `.mar` file

`model-servers` defines a custom model handler for deploying PyTorch models using TorchServe. The handler contains functions and classes for model initialization, preprocessing, inference, and postprocessing that can be switched out by for custom models and used for creating a `.mar` without much of the overhead required to package your model. We also provide helper scripts that can be used to package your model by simply refactoring your model source code.

The main components to package any pytorch model is as follows:
- **An initialisation function:** This function should return a torch.nn.Module object that represents your model object.
- **Model checkpoint**: This is your models `state_dict` that contains the models weights. Refer [here](https://pytorch.org/tutorials/beginner/saving_loading_models.html#what-is-a-state-dict) for what a `state_dict` is and how to save your model weights.
- **A requirements file**: This file contains additional python modules that you model requires. This is usually in the form of a `requirements.txt` file containing all the pakcages your model uses (Excluding torch/torchvision/torchaudio).

Once these three components are provided, we can swap out the appropriate modules in our custom handler with those for your model and package it into a `.mar`.

The `model-servers` handler takes care of certain things that users generally have to handle themselve by creating a custom handler. These include:
- Writing custom handlers for any new model that need to be packaged.
- Device assignment and management.
- Limiting memory for a particular model
- Server side request batching

We standardise certain aspects of torchserve handlers and make them reusable without the need to create a custom handler and add them separately for each new model.

## 3. Refactoring your model inference code
To be able to use the `model-servers` handler you must refactor your code into a certain format defined below. We recommend creating a separate file where you can import your model code and create the functions needed by the handler. This is useful to make changes to your model without the need to modify the handler.

### 1. Model Initialization Function
The `init_function` should be a function that initializes your PyTorch model. This function will be called during model loading. It should return an instance of your PyTorch model (an instance of `torch.nn.Module`). The handler allows you to pass additional arguments and keyword arguments to this function.

Let us suppose we define our model in a file called `model.py` (this file along with any other source code files are generally passed in the `extra-files` parameter to the `.mar`):
```python
import torch

class MyNetworkBlock(torch.nn.Module):
    def __init__(self, in_dim, out_dim):
        super().__init__()
        self.lin = torch.nn.Linear(in_dim, out_dim)
    def forward(self, x):
        x = self.lin(x)
        x = torch.relu(x)
        return x


class MyNetwork(torch.nn.Module):
    def __init__(self, in_dim, layer_dims):
        super().__init__()
        prev_dim = in_dim
        for i, dim in enumerate(layer_dims):
            setattr(self, f'layer{i}', MyNetworkBlock(prev_dim, dim))
            prev_dim = dim
        self.num_layers = len(layer_dims)
        self.output_proj = torch.nn.Linear(layer_dims[-1], 10)

    def forward(self, x):
        for i in range(self.num_layers):
            x = getattr(self, f'layer{i}')(x)
        return self.output_proj(x)
```

Our init_function is declared in a separate file, for example, `handler_model.py` as follows:
```python
from model import MyNetwork

def init_function(in_dim, layer_dims):
    model = MyNetwork(in_dim, layer_dims)
    return model

```

**NOTE**: The `init_function` can have any number of arbitrary parameters passed to it. These are passed as keyword arguments to the `init_function` by the handler. It should also be noted that the model is loaded without pre-trained weights. These waits are loaded automatically by the handler using the model checkpoint passed to it as described earlier.

### 2. Helper functions
We provide functionality to define pre-processing and post-processing functions that may be needed to modify the inputs to the model and the outputs from the model. These are simple python functions that need to be designed to process one sample of data (Since batching is handled by the handler on the server side). These are optional and can be left undefined, in which case the input data sent to the server is passed to the model as is. There are two functions that can be defined:

- Preprocessing Function (preprocess_function, optional): This function, if provided, preprocesses the input data before inference. It takes a single input and should return the preprocessed data.

- Postprocessing Function (postprocess_function, optional): This function, if provided, post-processes the model's output after inference. It takes a single input and should return the postprocessed data.

Here is a sample of a model code that uses both pre-processing and post-processing.

```python
from torchvision.models import resnet18, resnet50, ResNet18_Weights, ResNet50_Weights

def init_function(resnet_layers=18):
    # Torchvision models are instances of torch.nn.Module
    if resnet_layers == 18:
        resnet = resnet18(weights=None)
    elif resnet_layers == 50:
        resnet = resnet50(weights=None)
    return resnet


def preprocess(data):
    transforms = ResNet18_Weights.DEFAULT.transforms()
    return transforms(data)


def postprocess(data):
    prediction = data.squeeze(0).softmax(0)
    class_id = prediction.argmax().item()
    score = prediction[class_id].item()
    category_name = ResNet18_Weights.DEFAULT.meta["categories"][class_id]
    return score, class_id, category_name
```

In addition to these, two other functions are available:

- Install Packages Function (install_packages, optional): If your model requires any external packages or dependencies that need to be installed, you can define this function to handle their installation.

- Unpack Dependencies Function (unpack_dependencies, optional): If your model requires any additional files or dependencies that need to be unpacked or prepared, you can define this function to handle the unpacking process.

An example of these is given below:
```python
import os
# Alternatively you can use subprocess

def unpack_dependencies():
    os.system("wget https://someserver.com/somefile.zip")
    os.system("unzip somefile.zip")


def install_packages():
    os.system("git clone https://github.com/someproject/project.git")
    os.system("project/build.sh")
```

**NOTE:** `unpack_dependencies` need not be used for model source code files. These can be compressed into a `.tar.gz` file and passed to the handler and will be uncompressed by the handler. Any `pip` packages also need to be installed in `install_packages`, they can be added to the `requirements.txt` file and will be installed by torchserve before loading the model.

### 3. Creating the handler

Once all the functions have been defined, they can be added to a file and a handler class can be created using `base_handler`. [`base/base_handler.py`]

```python
import os
from torchvision.models import resnet18, resnet50, ResNet18_Weights, ResNet50_Weights
from base_handler import handler_factory

def init_function(resnet_layers=18):
    # Function should return a torch.nn.module
    if resnet_layers == 18:
        resnet = resnet18(weights=None)
    elif resnet_layers == 50:
        resnet = resnet50(weights=None)
    return resnet


def preprocess(data):
    transforms = ResNet18_Weights.DEFAULT.transforms()
    return transforms(data)


def postprocess(data):
    prediction = data.squeeze(0).softmax(0)
    class_id = prediction.argmax().item()
    score = prediction[class_id].item()
    category_name = ResNet18_Weights.DEFAULT.meta["categories"][class_id]
    return score, class_id, category_name


def unpack_dependencies():
    os.system("wget https://someserver.com/somefile.zip")
    os.system("unzip somefile.zip")


def install_packages():
    os.system("git clone https://github.com/someproject/project.git")
    os.system("project/build.sh")


handler = handler_factory(init_function=init_function,
                          preprocess_function=preprocess,
                          postprocess_function=postprocess,
                          install_packages=install_packages,
                          unpack_dependencies=unpack_dependencies,
                          resnet_layers=18)
```

**NOTE:** Notice how extra parameters needed to initialise the model are passed as keyword arguments to the `handler_factory`. This is limited for now to the `init_function`, we will later add options to pass arguments to all the other functions as well.

## 4. Packaging your handler
### Creating the MAR file

Once this model file is ready, we can use the build script [`base/build.sh`] to build the `.mar` using this model file.

The usage of the build script is as follows:
```
./build.sh --model_name <model_name> --model_file <model_file> --config_file
<config_file> --requirements_file <requirements_file> --checkpoint_file <checkpoint_file> --version <version> [--extra_files <extra_files>]"
```

The parameters are defined as follows:
- `model_name`: This is a unique name given to the model. TorchServe hosts the model at an endpoint based on this name. (`https://torchservehost/models/<model_name>`)
- `model_file`: This is the model file we defined in the previous section that contains the `init_function`, `preprocess_function`, etc. and the handler instance.
- `config_file`: This is model-config file of the model. This file can be used to define extra model parameters. This is the model-config defined by [TorchServe](https://github.com/pytorch/serve/tree/master/model-archiver#config-file)
We will define extra parameters that can be passed via this config file to the base_handler to control certain aspects of your model in the next section.
- `requirements_file`: This is the `requirements.txt` file that list all the python packages needed by your model. These are installed before your model is loaded.
- `checkpoint_file`: This is your model's saved `state_dict`, usually a `.pt` or `.pth` file.
- `version`: This defines your models version, usually `1.0`, `2.0`... etc. TorchServe uses this version to differentiate between different instances of the same model. (`https://torchservehost/models/<model_name>/<version>`)
- `extra_files`: This is any extra source code files that your `model_file` needs compressed into a `.tar.gz` file. These are unpacked in the same folder as `model_file` maintaining the same directory structure they were packed with. This parameter also needs to be added to the model-config as defined in the next section. Please use `tar -cvf <extra_files> -C file1 file2 directory/` to compress your files.


### Model config

We explain the model-config and its parameters below. This file is defined by TorchServe [link](https://github.com/pytorch/serve/tree/master/model-archiver#config-file)
We add a few extra parameters to control things like model memory limit, and the source code files. Here is a sample model-config

```yaml
# TorchServe controls. These parameters are used to control model workers, batching, etc. These parameters are optional, if not provided TorchServe will use default values.

minWorkers: 1   # Minimumum number of workers needed for the model

maxWorkers: 1   # Maximum number of workeres needed for the model, this is the upper limit that TorchServe will scale your model.

maxBatchDelay: 4000     # This is the maximum delay between requests that TorchServe uses to batch requests(in msec). Any requests received within this time limit will be batched upto the batchSize.

responseTimeout: 5200   # This is the maximum delay before TorchServe returns a failed reponse. Please set this appropriately
batchSize: 4    # The batchsize of the model.

deviceType: "gpu"   # This is the device that the model needs to be placed on. [cpu, gpu, neuron]. In case of multiple GPUs on the system, the device is assigned automatically by TorchServe.

deviceIds: [0,1,2,3] # gpu device ids allocated to this model.

# These are extra parameters that we define for our model-servers handler.
system:
  extraFiles: "test.tar.gz"     # This is the extra source code files, adding them here tells the handler what file to unpack.

  maxGPUMemory: 500     # Max GPU Memory needed by model in MB. In case this is set, the model will not be able to access more than the limit.
```
