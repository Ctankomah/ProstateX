## prostate-xai
This contains all code related to ProstateX AI, an ensemble of deep learning models for prostate cancer classification.

NB: You need to have Node JS and pgAdmin installed. 


 ## Express Server
 This contains the views(ejs) for uploading and displaying results to the user and the nodejs
 code for sending requests to the python code for analysis

 ## model backend

 This contains a flask app that handles requests from the express server.
 It has a /process-zip route that takes a zip file and passes it to the model for analysis.

 It then returns the results after the model runs
