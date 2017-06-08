pipeline {
    agent any
    stages {
        stage('Build') {
            steps {
                script {
                    sh './00_build_deploy.sh'
                }
            }
        }
        stage('Start Cluster') {
            steps {
                script {
                    sh "./01_start_cluster.sh brave-monitor-160414 "
                }
            }
        }
        stage('Run PreNN processor') {
            steps {
                script {
                    cluster = readFile '.cluster'
                    sh "./02_apply_categories_extract_text.sh "
                }
            }
        }
    }
}