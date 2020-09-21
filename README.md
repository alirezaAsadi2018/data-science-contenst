# data-science-contenst
# question 1 (sales forecast):
One of our constant challenges is to forecast sales of different products at different time intervals. This prediction is important in several ways; The most important of these are helping sellers to have different products available, which will maximize profitability to the seller goods, and help not to delay the delivery of customer orders. In this case, the sales data of the last 5 years will be provided to you for 50 different sellers. You can download this data from the link below. (data is available in q1 directory!)

This archive file has 3 data files. The first file is train.csv , which contains the training instruction data. This file has the following columns.  

- id: Line ID
- date: record date
- seller: seller
- item: Product ID
- sales: The number of products sold  

The second file is test.csv , which contains the rows that should predict sales. This file has columns like the training file and only the sales column is not there and you have to predict it.

The third file is answer_style.csv , which indicates the file format you need to upload. For more simplicity, we also explain the format below.  
### Sample output 1
```
id,sales
0,100
1,100
2,100
3,100
4,100
5,100
```
This file has 2 columns. Which should be filled as shown above. The numbers in both columns must be correct and all the ids in the test file must be present in this file and their sales value must be present so that your answer can be judged. You must upload the completed file in csv format.

The SMAPE criterion is used to measure the quality of your answer according to the problem .

# question 2 (approving or rejecting comments):
We offers users a diverse range of goods with a variety of categories. Users can share their opinions and experiences about each product with others. These comments are placed on the product page and are visible to all users. We filter comments with inappropriate or irrelevant content. The process of approving or rejecting comments used to be done by human resources. In this challenge, you have to design a model by which you can reject or approve comments.
In this case, two files are provided to you. You can download these files from the link below.

(The training file on this issue and the issue of predicting the rating of opinions are the same) (data is available in q2 directory!)
The first file available to you is the train_users.csv file . Use this file to teach your model. This file has the following columns:  
- id : The ID of each comment
- title : Comment title
- comment : Comment text
- advantages : The benefits mentioned in the comment
- disadvantages : The disadvantages mentioned in the comment
- title_fa_product : The Persian name of the product
- title_fa_category : Persian name of the product category
- is_buyer : The buyer of the product or not the commenting user
- verification_status : Reject and approve comments
- rate : The rating given to the product in question  

The id column is the unique identifier of each comment. The title column is the title that the user wrote for the comment. The comment column is the text of each comment. The advantages column is the user listed benefits for the product. The disadvantages column is the user listed disadvantages for the product. The title_fa_product column is the Persian name of the product. The title_fa_category column is the Persian name of the product category. The verification_status column indicates that the comment has been rejected or approved. If the comment is approved, it is 1 and if it is rejected, it is 0. The rate column is the corresponding score for each comment, which is a number between 0 and 100.  

The second file you have is the test_users.csv file . This is a test data file that you should earn by scoring to confirm or reject the match. This file has the same columns above except verification_status and rate .  
You must predict for the test data that the comment has been approved or rejected. Your output to be uploaded should look like this:  
### Sample output
```
id,verification_status
0,0
1,1
2,1
3,0
4,1
5,1
6,1
7,0
…
```
This file has two columns id and verification_status . The comment id of this file is the same as the comment id of the test file, which must be in the same order and number in this file. The value of verification_status in this file must be a valid form with values ​​of 0 or 1.

In this case, only your output file will be sent for judging. So you are free to solve the problem with any programming language and any method.

The score criterion in this issue is your output f1 score.

# question 3 (Predict comments rating):
We offer users a diverse range of goods with a variety of categories. Users can share their opinions and experiences about each product with others. These comments are placed on the product page and are visible to all users. One of the information that a user can share with others is the overall rating of the product. This score is usually directly related to the text written by the user. In this case, you are asked to use the comments to predict the user rating of the product.  
In this case, two files are provided to you. You can download these files from the link below.(data is available in q3 directory!)  
(The training file is the same in this problem and problem 2: **approving or rejecting comments**)  
The first file available to you is the train_users.csv file . Use this file to teach your model. This file has the following columns:  
- id : The ID of each comment
- title : Comment title
- comment : Comment text
- advantages : The benefits mentioned in the comment
- disadvantages : The disadvantages mentioned in the comment
- title_fa_product : The Persian name of the product
- title_fa_category : Persian name of the product category
- is_buyer : The buyer of the product or not the commenting user
- verification_status : Reject and approve comments
- rate : The rating given to the product in question  

The id column is the unique identifier of each comment. The title column is the title that the user wrote for the comment. The comment column is the text of each comment. The advantages column is the user listed benefits for the product. The disadvantages column is the user listed disadvantages for the product. The title_fa_product column is the Persian name of the product. The title_fa_category column is the Persian name of the product category. The verification_status column indicates that the comment has been rejected or approved. If the comment is approved, it is 1 and if it is rejected, it is 0. The rate column is the corresponding score for each comment, which is a number between 0 and 100.  

The second file you have is the test_users.csv file . This is the problem test data file that you get the question score by predicting the comment score. This file has the same columns above except for verification_status and rate .

You must predict the comment score for the test data. Your output to be uploaded should look like this:  
### Sample output
```
id,rate
0,57.45175680275809
1,88.08661090571425
2,81.00113574934842
3,46.017899851519914
4,32.7560294359769
5,37.174967371155866
…
```
This file has two columns id and rate . The comment id of this file is the same as the comment id of the test file, which must be in the same order and number in this file. The rate value in this file must be a decimal number with values from 0 to 100.

In this case, only your output file will be sent for judging. So you are free to solve the problem with any programming language and any method.

The score criterion in this issue is SMAPE, which will be calculated based on the scores you predict.

# question 4 (image processing/ Recognize the color of photos):
Easy access to the desired products is a constant challenge. One way to reach the desired product is to use search. But sometimes finding products by searching is not easy. For this reason, it is possible to filter search results. One of the useful filters, especially in the field of fashion and clothing, is the color filter of the product, but sometimes some products lack color information. In these cases, it is tried to use the product information to identify the color of the product and to filter it in color.  
In this issue, we ask you to identify the color of the products provided to you using the photo.  
You can download the required data of the issue from the following link.(data is available in q4 directory!)  
This file contains 2 folders, train and test . In the train folder there is a photo gallery required for color separation.  
In the test folder there are files that you must use your algorithm to identify their color and upload this question to get points.  
The file format uploaded by you should look like this:  
### Sample output
```
file_name,color_id
117142451.jpg,2
117221672.jpg,1
117210354.jpg,1
117129967.jpg,4
117304479.jpg,11
117238605.jpg,6
117242357.jpg,9
```
Note that the number of rows in your file must be equal to the number of files in the test file, and the file names must be given in full. In the color_id column, enter the detected color ID. You can use the table below to convert color to ID.  
```
color,color_id
pink,1
purple,2
yellow,3
orange,4
white,5
silver,6
grey,7
black,8
red,9
brown,10
green,11
blue,12
```
The numeric ID is between 1 and 12 and must be correctly placed in your output file.

In this case, only your output file will be sent for judging. So you are free to solve the problem with any programming language and any method.

The score criterion in this issue is precision , which will be calculated from the colors recognized by you.

# question 5 (database sqlite problem):
One of the criteria for user behavior on the site is to find out how active the user is. Suppose user login and logout data is available on the site as shown below.
```
user_id       date            status 
  101        2018-01-1          1 
  101        2018-01-2          0 
  101        2018-01-3          1   
  101        2018-01-4          1 
  101        2018-01-5          1
```
In this case, you must use this data to be able to calculate the level of activity of each user in each period.  
To solve this problem, the following data is provided to you.(data is available in q5 directory!)  
This file provides a sqlite database called dkcup_test . Also the output that was given to you in a file called answer_test_db_users.csv by running the query correctly.  
In this case, you must upload your query to the site as a sql file (such as query.sql ).  
The sample output should be as follows.  
```
user_id  start_date     end_date  status  length 
  101    2018-01-1    2018-01-1    1      1
  101    2018-01-2    2018-01-2    0      1
  101    2018-01-3    2018-01-5    1      3
```
Also note that all columns must be completely present in your submission file and their names must be exactly the same as the example above and the response file provided to you.  
You can check the correctness or inaccuracy of your query with the answer file and database provided to you. Due to the large size of the database and the weight of its calculations, we recommend using a smaller part of the database file for testing.  
This question also uses version 3.25 of the sqlite database .

# question 6 (c++ problem):
A large warehouse consists of a number of product tank. These tanks are connected to a network of conveyor belts to transport goods to various order processing units. The conveyor belt network is finally connected to the warehouse output terminals to load and send orders.  
The direction of rotation of the belts is one-way. The beginning of each belt is a tank(source) or a processing unit, and the end is a processing unit or an output terminal(sink).  
Each belt has a specific capacity to transport goods. Processing units (and tanks and terminals) have unlimited capacity. Their only condition is that the total number of goods entering them with the belts is equal to the total number of output goods (because the goods are not to appear or disappear during processing!).  
In the first input line you are given the number of storage units (n), including tank, intermediate processing unit or terminal.  
In the second line, you are given **n** numbers, and the **i**th number  indicates the type of unit i. If this number is zero, it means that the corresponding unit is the processing unit. If it is 1 or 2, it means that it is the unit of the tank or output terminal, respectively.  
The next line shows the number of belts (e). In each of the next **e** lines, the number of the beginning unit, end unit, and the capacity of one of the belts is given.  
In response, print the maximum number of goods that can be processed per day in one line. The maximum number of goods that can be processed is the maximum number of goods that can be delivered from the tanks to the terminals according to the capacity of the belts and the conditions mentioned.
  ### Sample input
  ```
  7
  1 0 0 0 0 2 1
  10
  1 2 16
  7 3 13
  2 3 10
  3 2 4
  2 4 12
  4 3 9
  3 5 14
  5 4 7
  4 6 20
  5 6 4
  ```
  ### Sample output
  ```
  23
  ```
