pipeline {
  agent any

  environment {
    PYTEST_ARGS = '--cov=daceml --cov-report term --cov-report xml --cov-config=.coveragerc --gpu'
    PYTHON = '/usr/bin/python3'
    CUDA_ROOT = '/usr/local/cuda'
    ORT_ROOT = '/home/orausch/onnxruntime'
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
