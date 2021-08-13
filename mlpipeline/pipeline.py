import logging
import json
import kfp
from kfp.aws import use_aws_secret
import kfp.dsl as dsl
from kfp import components
import json

logging.basicConfig(level=logging.INFO, format='%(name)s - %(levelname)s - %(message)s')


def create_preprocess_component():
    component = kfp.components.load_component_from_file('./components/preprocess/component.yaml')
    return component

def create_train_component():
    component = kfp.components.load_component_from_file('./components/train/component.yaml')
    return component

def mlpipeline():
    preprocess_component = create_preprocess_component()
    preprocess_task = preprocess_component()

    train_component = create_train_component()
    train_task = train_component(data_path=preprocess_task.outputs['output_path'])

    deployment_op = components.load_component_from_url('https://raw.githubusercontent.com/kubeflow/pipelines/master/components/kubeflow/kfserving/component.yaml')
    container_spec = {"image": "thanhhau097/ocrdeployment", "port":5000, "name": "custom-container", "command": "python3 /src/main.py --model_s3_path {}".format(train_task.outputs['output_path'])}
    container_spec = json.dumps(container_spec)
    deployment_op(
        action='apply',
        model_name='custom-simple',
        custom_model_spec=container_spec, 
        namespace='kubeflow-user',
        watch_timeout="1800"
    )

kfp.compiler.Compiler().compile(mlpipeline, 'mlpipeline.yaml')