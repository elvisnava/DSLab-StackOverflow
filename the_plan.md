# Implement Baselines

### Papers he gave us which are kinda different
* IEEE 2015: Exploiting User Feedback for Expert Finding in Community Question Answering
    * ```Question, Set[User] -> OrderedList[User]```, i.e. for a question with all the people that answered it they try to predict the ranking of the answers by upvotes
* CIKM 2010: ```User -> is_expert=True/False```


## ```User -> List[Questions]```
* ACM 2015: "Predicting answering behaviour in online question answering communities." 

##  ```Question -> List[User]```
* WWW 2012: Finding Expert Users in Community Question Answering
    * calculate user/question embedding with tf-idf, LDA, STM 
    * sort users by simiarity => could also use the other way around

* ACM 2015: Who will Answer my Question on Stack Overflow? 
    * Technically they do ```Question -> Set[Users]```, since they binary classify `(question,user)` pairs