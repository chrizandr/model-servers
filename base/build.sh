#!/bin/bash

# Function to display usage instructions
function display_usage {
    echo "Usage: $0 --model_name <model_name> --model_file <model_file> --config_file <config_file> --requirements_file <requirements_file> --checkpoint_file <checkpoint_file> --version <version> [--extra_files <extra_files>]"
}

# Parse named arguments
while [[ $# -gt 0 ]]; do
    case "$1" in
        --model_name)
            model_name=$2
            shift 2
            ;;
        --model_file)
            model_file=$2
            shift 2
            ;;
        --config_file)
            config_file=$2
            shift 2
            ;;
        --requirements_file)
            requirements_file=$2
            shift 2
            ;;
        --checkpoint_file)
            checkpoint_file=$2
            shift 2
            ;;
        --version)
            version=$2
            shift 2
            ;;
        --extra_files)
            extra_files=$2
            shift 2
            ;;
        *)
            display_usage
            exit 1
            ;;
    esac
done

# Check if all required arguments are provided
if [ -z "$model_name" ] || [ -z "$model_file" ] || [ -z "$config_file" ] || [ -z "$requirements_file" ] || [ -z "$checkpoint_file" ] || [ -z "$version" ]; then
    echo "Error: All arguments must be provided"
    display_usage
    exit 1
fi

# Your script logic goes here
echo "Building MAR with the following configuration....."
echo "-------------------"
echo "TorchServe model / MAR file name: $model_name"
echo "Model / MAR version: $version"
echo "Model code file: $model_file"
echo "TorchServe model config file: $config_file"
echo "Model requirements file: $requirements_file"
echo "Model checkpoint file: $checkpoint_file"

# temp_dir=$(mktemp -d)
# random_filename=$(mktemp --tmpdir="$temp_dir" XXXXXX.py)
# cp $model_file handler_model.py 2>/dev/null

if [ -n "$extra_files" ]; then
    echo "Extra files: $extra_files"
    echo "-------------------"
    echo "Copying model file to: handler_model.py"
    torch-model-archiver -f --model-name $model_name --version $version \
                     --handler $model_file \
                     --serialized-file $checkpoint_file \
                     --runtime python3 \
                     --requirements-file $requirements_file \
                     --model-file base_handler.py \
                     --config-file $config_file \
                     --extra-files $extra_files
else
    echo "-------------------"
    echo "Copying model file to: handler_model.py"
    torch-model-archiver -f --model-name $model_name --version $version \
                     --handler $model_file \
                     --serialized-file $checkpoint_file \
                     --runtime python3 \
                     --requirements-file $requirements_file \
                     --model-file base_handler.py \
                     --config-file $config_file
fi

# ./build.sh --model_name test-model --version 1 --model_file handler_model.py \
#            --config_file model-config.yml --requirements_file requirements.txt \
#            --checkpoint_file /data/chris/model-servers/models/resnet/resnet18-f37072fd.pth \
#            --extra_files test.tar.gz

# sudo cp test-model.mar /data/docker/volumes/ts-test/_data/pytorch-models
