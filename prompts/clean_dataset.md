Your task is to read the csv file present in ./datasets/uncleaned/000.csv and you have to do the following task, for every row present in a csv, you have to generate multiple input text such that some text should use the values present 
in input_parameters, in it's input text and when not using the input_parameters in it's input text the bash command to replace it's needed arguments with placeholders. 

Here are couple of examples:


1. git search commit messages, "[""fix bug""]","git log --grep=""fix bug"""

Newly generated rows and their corresponding bash commands

- git search for commit message "fix bug", "git log --grep=""fix bug"""
- git search for commit message fix bug, "git log --grep=""fix bug"""
- git search for commit message, "git log --grep=""YOUR_COMMIT_MESSAGE"""


2. docker run a container,"[""nginx""]",docker run -d nginx

Newly generated rows and their corresponding bash commands

- docker run nginx container, docker run -d nginx
- docker run container, docker run -d YOUR_DOCKER_CONTAINER


You should create a new csv file 000_cleaned.csv, with no input_parameters column.

You have to do it yourself manually row by row, i.e. read each row and then manually generate new rows  
