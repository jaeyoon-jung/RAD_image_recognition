# RAD_image_recognition
Rotten Apple Detector (RAD)- Project for Criteo's Global Hackathon; a machine learning model with 
deep learned features that can detect inappropriate images. 

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


## Main Environment Prerequisite
* Python 3.x
* Scikit-learn
* XGBoost
* Tensorflow 


## Methodology
To build an image recognition model that can be trained to detect bad products, I used a Convolutional Neural Network (CNN) as
a main algorithm to idenitfy image features. Specifically, I applied transfer learning on Inception V3 architecture, 
which is an idea of preserving the architecture of a pre-trained model but re-training it on a new dataset with desired
classes. Inception V3 model here is a type of CNN model that was trained on Imagenet dataset, a standard academic dataset with 
1000 classes commonly used for training an image recognition model. In the original model, the layer right before the final 
classification layer returns 2048 dimension vector, which essentially represents 2048 features of an image. 
Here, instead of using a origina finaly layer, I added a layer of XGBoost training, which then uses these 2048 deep-learned 
numeric features and classes of the new dataset (product/inappropriate iamges) to fit an XGBoost model. 

Although transfer learning is not as accurate as a fully trained CNN architecture, it is generally known to achieve reasonable accuracy. This method was also appropriate for this Hackathon, given the following conditions:

1. As an analyst of a Criteo's regional office (Tokyo hub) I did not have access to data infrastructure that the R&D team in Paris does (at least during the Hackathon). The model would have to be redesigned & retrained with a refined dataset anyway.
2. Even with a perfect training dataset, my laptop alone might not be able to handle a complete training without GPU.
3. I had 48 hours for this Hackahton. 
4. Proof of concept, rather than a fully developed solution, would be sufficient for pitch & demo.


## Data
As a proof of concept, instead of training to detect all possible bad images, I chose three sample 'bad' categories:

1. Gun: some publishers do not allow display of weapons
2. Knife: same as above
3. Nudity: Criteo as a whole does not allow any sexual images. In this context, I defined nudity as very revealing images in general, not strict nudity. 

Then the model was trained to classify images into 4 categories: Good Product, Gun, Knife, and Nudity. Data for bad categories
were collected by batch-downloading 400 images from Google per class. For good products, I downloaded 400 images from Criteo's
internal resource, to include a variety of products. 


## Result:
When validated with K-Fold cross validation with 5 folds, the model achieved approximately 95.5% accuracy. 

Below is a confusion matrix of the 5-fold validation:


For the demo, I demonstrated the performance of the model using real product images from Criteo's feed. Out of 3 good products, 2 knives, 2 nudity, and 2 guns, the model classified 8/9 images correctly. Because of constraints on time and
data access, I could not create an additional test dataset that strictly consists of Criteo's product images. 
