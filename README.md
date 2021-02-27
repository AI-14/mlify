# MLify
  ![Python](https://img.shields.io/badge/-Python-black?style=flat&logo=python)
  ![Machine Learning](https://img.shields.io/badge/-Machine%20Learning-566be8?style=flat)
  ![Sklearn](https://img.shields.io/badge/-Sklearn-1fb30e?style=flat)
  ![Streamlit](https://img.shields.io/badge/-Streamlit-f0806c?style=flat)
  ![Heroku](https://img.shields.io/badge/-Heroku-6174c2?style=flat&logo=heroku)

## Description
   MLify is a simple semi-autoML web application designed for those who are beginners in the field of machine learning and have no coding experience. With this app, they can 
   now perform EDA, visualize graphs/charts, and run over 5 different ML algorithms on various datasets that we have provided. There is also an option of automated EDA - which
   can save a lot of time. Furthermore, it can also be used for educational purposes by university instructors. The app is deployed on heroku platform and to check it out, click
   on the website link under *Web App Link*.
   
## Web App Link
[MLify Web App Link](https://mlify.herokuapp.com/)

## Screenshots Of The App
<details>
  <summary>Click to expand!</summary>

![](/res/readme_res/Pic1.png)

![](/res/readme_res/Pic2.png)

![](/res/readme_res/Pic3.png)
</details>

## Installation And Usage
<details>
  <summary>Click to expand!</summary>
  
1. Installation
   - Download/clone this repository. Then open terminal (make sure you are in the project's directory).
   - Create a virtual environment using the command ````py -m venv yourVenvName```` and activate it using ````yourVenvName\Scripts\activate.bat````.
   - Then run the following command ````pip install -r requirements.txt````. With this, all the dependencies will be installed in your virtual environment. 
> **Note:** *If any dependency is missing or an error shows up, install it using ````pip install moduleName````*.

2. Usage
   - Open your project folder and go to the terminal and activate your virtual environment. Then type ````streamlit run src\main.py```` and the app will open in your web 
   browser. Now you can interact with it or play with the code and add your own features and if you wish - you can deploy your own version of MLify on heroku.

>**Note:** When specifying the path to images or any other resource in the source code - please note that the slashes used can be different based on the IDE/TextEditor you are using. Hence, refactor accordingly.
</details>
