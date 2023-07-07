# Zomato Bangalore Restaurants

The basic idea of analyzing the Zomato dataset is to get a fair idea about the factors affecting the establishment
of different types of restaurant at different places in Bengaluru, aggregate rating of each restaurant, Bengaluru
being one such city has more than 12,000 restaurants with restaurants serving dishes from all over the world.
With each day new restaurants opening the industry has’nt been saturated yet and the demand is increasing
day by day. Inspite of increasing demand it however has become difficult for new restaurants to compete with
established restaurants. 

## Dataset 


[Link to Zomato kaggle Dataset ](https://www.kaggle.com/datasets/himanshupoddar/zomato-bangalore-restaurants)


## Run Locally

Clone the project

```bash
  git clone https://github.com/amaithi-sam/zomato_ml.git
```

Go to the project directory

```bash
  cd zomato_ml
```

Install dependencies

```bash
  pip install -r requirements.txt
```

Run the Project

```bash
  python dvc repro -f
```


# mlflow ui



### if want to use the model registry feature, we need a database.

#### _If you have MySQL installed then you can use the below command:_

1. Create a database to use as an MLflow backend tracking server.

`CREATE DATABASE mlflow_tracking_database;`

2. Start MLflow tracking server using MySQL as a backend tracking store.

`mlflow server \
   --backend-store-uri  mysql+pymysql://root@localhost/mlflow_tracking_database \ 
   --default-artifact-root  file:/./mlruns \
   -h 0.0.0.0 -p 5000`


3. Set the MLflow tracking uri (within code section).

  mlflow.set_tracking_uri("http://localhost:5000")

#### _If you have sqlite installed then you can use the below command:_

1. Start MLflow tracking server using sqlite as a backend tracking store.

`mlflow server --backend-store-uri sqlite:///mlflow.db --default-artifact-root ./artifacts --host 0.0.0.0 --port 5001`


2. Set the MLflow tracking uri (within code section).
    
    `mlflow.set_tracking_uri("http://localhost:5001")`


You can also follow the official documentation for more information on backend database for model registry

https://www.mlflow.org/docs/latest/model-registry.html#model-registry-workflows







## Author

- [@amaithi-sam](https://www.github.com/amaithi-sam)


