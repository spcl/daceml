pipeline {
  agent any

  environment {
    CODECOV_TOKEN = credentials('codecov-token')
    PYTEST_ARGS = '--cov=daceml --cov-report term --cov-report xml --cov-config=.coveragerc --gpu'
    PYTHON = '/usr/bin/python3'
    CUDA_ROOT = '/usr/local/cuda'
    ORT_ROOT = '/home/rauscho/onnxruntime'
    TORCH_VERSION = 'torch==1.6.0+cu101 torchvision==0.7.0+cu101 -f https://download.pytorch.org/whl/torch_stable.html'
  }

  stages {
    stage('Setup') {
      steps {
        sh 'make clean install'
      }
    }
    
    stage ('Test') {
      steps {
        sh '''
	LD_LIBRARY_PATH=/usr/local/cuda/lib64:$LD_LIBRARY_PATH
	PATH=/usr/local/cuda/bin:$PATH
        make test codecov check-formatting
        '''
      }
    }
  }
}
