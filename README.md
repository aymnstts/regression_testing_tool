# regression_testing_tool
you only need python installed in your computer  for this project to run locally,download it using this link:
windows:
https://www.python.org/downloads/
mac:
https://www.python.org/downloads/macos/
once you unzip the rar or clone the repo:
verify that you are in /projet directory 
1. Create and activate a virtual environment

It’s always best to work in an isolated Python environment.

# 1- Create a virtual environment
python -m venv venv

# Activate it
# On Windows:
venv\Scripts\activate
# On macOS/Linux:
source venv/bin/activate


#  2- Install Flask and required libraries
pip install flask flask-cors werkzeug pandas openpyxl numpy


# 3-  run
python app.py

# notes

once you run the project,you can upload the files needed to be tested,the file name of the file tested should be without the any dates for example: Cash_Sub not Cash_Sub291019
Also if you added a new reference file,make sure the name is also in the same manner -> Reference_File.

if you have any new reference file upload it before testing.

ps: all requirements are also in the requirements.txt

