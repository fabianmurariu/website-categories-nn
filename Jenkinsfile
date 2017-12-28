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
                    sh "./02_1_top3_categories_extract_text.sh "
                }
            }
        }
        stage('Run PreNN tokenizer') {
            steps {
                script {
                    sh "./03_transform_to_features.sh "
                }
            }
        }
        stage('Shutdown Cluster') {
            steps {
                script {
                    sh "./04_shutdown_cluster.sh "
                }
            }
        }
    }
}