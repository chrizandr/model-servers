model_name=test-model

torch-model-archiver -f --model-name $model_name --version 1.0 \
                     --handler cai_handler.py \
                     --serialized-file /data/chris/model-servers/models/resnet/resnet18-f37072fd.pth \
                     --runtime python3 \
                     --requirements-file requirements.txt \
                     --model-file test_model.py \
                     --config-file model-config.yml

sudo cp $model_name.mar /data/docker/volumes/ts-test/_data/pytorch-models/
