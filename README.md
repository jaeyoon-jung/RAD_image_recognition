# RAD_image_recognition
Rotten Apple Detector (RAD)- Project for Criteo's Global Hackathon; a machine learning model with 
deep learned features that can detect inappropriate images

## Rationale
When advertisers include products with inappropriate descriptions or images in their feed, 
they end up getting blacklisted by publishers. These incidents stop displaying
these advertisers' performance and damange Criteo's relationship with other publishers, eventually
costing the company a large amount of revenue. For example, the analysis of publisher's share of 
voice and advertiser revenue estimates that Criteo can gain a potential revenue uplift of 4.5% and 6.3% in 
Japan and China respectively, if it can resolve all blacklisting issues. 

Currently, the responsibilities of keeping bad products out of feed are all on advertisers, and there is no internal tool
within Criteo's ecosystem to detect one. The objective of this project is to develop a solution to blacklisting 
problems through automatic detection of bad (inappropriate) images.


## Methodology

To build an image recognition model that can be trained to bad products, I used a Convolutional Neural Network. 
