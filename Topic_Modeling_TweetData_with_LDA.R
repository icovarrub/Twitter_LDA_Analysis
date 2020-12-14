##################################################################################################################################
######################################### WEB EXERCISE 8 - TOPIC MODELING WITH LDA ###############################################
##################################################################################################################################

#Install packages necessary to executve this exercise
install.packages(c("plyr", "stringr","tm", "SnowballC", "lda", "LDAvis"))

library(plyr)
library(stringr)
library(tm)
library(SnowballC)
library(lda)
library(LDAvis)

#set working directory
setwd("~/R")

#load twitter csv file
Twitter <- read.csv("TwitterSample.csv")

######## CLEANING AND FORMATTING ########

#Create corpus with Tweet texts
Tweets_corpus <- Corpus(VectorSource(Twitter$TWEET_TEXT))

#Convert uppercase letters to lowercase using tolower function
Tweets_corpus <- tm_map(Tweets_corpus, tolower)
Tweets_corpus[[1]]

#Remove punctuation
Tweets_corpus <- tm_map(Tweets_corpus, removePunctuation)
Tweets_corpus[[1]]

#Remove numbers
Tweets_corpus <- tm_map(Tweets_corpus, removeNumbers)
Tweets_corpus[[1]]

#Remove URLs
Tweets_corpus <- tm_map(Tweets_corpus, function(x) gsub("http[[:alnum:]]*","",x))
Tweets_corpus[[1]]

#Remove NonASCII characters
Tweets_corpus <- tm_map(Tweets_corpus, function(x) iconv(x,"latin1","ASCII",sub = ""))
Tweets_corpus[[1]]

#remove specific words
Tweets_corpus <- tm_map(Tweets_corpus, removeWords, c("london", "im", "ive", "dont", "didnt"))
Tweets_corpus[[1]]

#Strip white spaces
Tweets_corpus <- tm_map(Tweets_corpus, stripWhitespace)
Tweets_corpus[[1]]

#stemming words to reduce to a words stem form
Tweets_corpus <- tm_map(Tweets_corpus, PlainTextDocument)
Tweets_corpus <- tm_map(Tweets_corpus, stemDocument)
Tweets_corpus[[1]]

###########Convert corpus back to character list and data frame
###########remove extra white spaces

#Unlist text corpus
Tweet_Clean <- as.data.frame(unlist(sapply(Tweets_corpus[[1]]$content,'[')), stringsAsFactors = F)

#remove extra white spaces in text
Tweet_Clean <- lapply(Tweet_Clean[,1], function(x) gsub("^","",x)) #multiple spaces
Tweet_Clean <- lapply(Tweet_Clean, function(x) gsub("^[[:space:]]+","",x))
Tweet_Clean <- lapply(Tweet_Clean, function(x) gsub("[[:space:]]+$","",x))

########### Compare changes with original

#bind clean text with Twitter Data
Twitter$Tweet_Clean <- Tweet_Clean

#check first 10 tweets
Twitter[1:10,]

######## PREP FOR TOPIC MODELING ########

#Tokenize on space and output as a list
doc.list <- strsplit(unlist(Tweet_Clean),"[[:space:]]+")

#########Compute frequency of each term
#########Remove low frequency terms
#########Doing this will alway next processes much easier

#Compute the table of terms
term.table <- table(unlist(doc.list))
term.table <- sort(term.table, decreasing = TRUE)

#Remove terms that are stopwords or occur fewer than 3 times
term.table <- term.table[term.table>3]

#Create a vocabulary with remaining terms
vocab <- names(term.table)

#########Put documents into the format required by lda
#########"Documents" is a list where each element represents 1 document (Tweet)

#Put documents into format required by lda package
get.terms <- function(x){
  index <- match(x,vocab)
  index <- index[!is.na(index)]
  rbind(as.integer(index - 1), as.integer(rep(1,length(index))))
}

documents <- lapply(doc.list, get.terms)

######## STATISTICS ON OUR DATA SET #########

#Compute some statistics related to the data set
D <- length(documents) #number of documents

W <- length(vocab) #number of terms in the vocab

doc.length <- sapply(documents, function(x) sum(x[2,])) #number of tokens per doc

N <- sum(doc.length) #total number of tokens in the data

term.frequency <- as.integer(term.table) #frequencies of terms in the corpus

########### FIT LDA MODEL ###############

#Parameters

K <- 20
G <- 1000
alpha <- 0.1
eta <- 0.1

t1 <- print(Sys.time())

lda.fit <- lda.collapsed.gibbs.sampler(documents=documents, K=K, vocab=vocab, num.iterations = G,alpha = alpha, eta = eta)

t2 <- print(Sys.time())

t2-t1

#Lets examine topics by function "top.topic.words"
#Will return a matrix of the top words in each topic
#Number 20 here means top 20 words of each topic will be shown

top_words <- top.topic.words(lda.fit$topics,20,by.score = TRUE)

top_words

#Run LDAvis package to visualize the results
#Interative visualization of the topics
#Document-topic distribution estimates needed (D*K matrix theta)
#Set of topic-term distributions needed (K*W matrix phi)

theta <- t(apply(lda.fit$document_sums + alpha, 2, function(x) x/sum(x)))

phi <- t(apply(t(lda.fit$topics) + eta, 2, function(x) x/sum(x)))

#With the stats, save theta, phi, and vocab in a list as the data object "Tweet_Topics"
Tweet_Topics <- list(phi=phi, theta=theta, doc.length=doc.length, vocab=vocab,
                     term.frequency=term.frequency)

#Use "createJSON() function to have a JSON object used for visualization
#Function will compute topic frequencies, inter-topic distances, and project topics onto a 2D plane to represent their similarity to each other

Tweet_Topics_json <- with(Tweet_Topics,createJSON(phi,theta,doc.length,vocab,term.frequency))

serVis(Tweet_Topics_json)

############# ASSIGN TOPIC FOR EACH TWEET ################

#Topic model can return probabilities of each word belonging to different topics
#Within a tweet, different words carry different topics
#For this, topic appearing most frequently within that Tweet will be the topic that the Tweet is assigned to

doc_topic <- apply(lda.fit$document_sums,2,function(x) which(x==max(x))[1])
Twitter$topic <- doc_topic

#Now each Tweet has its dominant topic
#Lets look at the spatial distribution of each topic
#Export objec "Twitter" and display it in ArcMap

TwitterDf <- data.frame(lapply(Twitter, as.character),stringsAsFactors = FALSE)
write.csv(TwitterDf,"TwitterWithTopic.csv")

#Visualize CSV file using Google Fusion Table