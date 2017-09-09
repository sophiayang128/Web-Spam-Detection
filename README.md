# Web-Spam-Detection
* This is a Web Spam Detection project. In this project, we combined content-based analysis together with link-based analysis into the web spam detection system. The dataset we used is 'WEBSPAM-UK2007', which is an highly imbalanced dataset. <br>
* We converted the imbalanced training dataset to multiple balanced subsets using random under-sampling, and trained the classifier based on those balanced subsets. <br>
* According to our experimental results, the combination of Gradient Boosting Decision Tree classifier and RUS-Multiple under-sampling method has the best performance, which gives an AUC score higher than the first place of Web Spam Challenge 2008.
