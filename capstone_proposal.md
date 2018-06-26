# Machine Learning Engineer Nanodegree
## Capstone Proposal
J N
June 24st, 2018

## Proposal
_(approx. 2-3 pages)_
The goal is to design and develop an App in Python that recognizes faces in real time. The App connects to any given camera device and displays the camera's content. Whenever the current camera image shows one (or multiple) faces they should be highlighted in the shown image. If the App recognizes any of those faces, the person's name should be displayed in the shown image.

On user input a new face can be recorded and added to the database in real time. Afterwards the App will be capable of recognizing the face as well.

The facial images that a similarity vector is calculated upon that is added to the database for recognition purposes are saved to a local directory. That way all calculations can be reconstructed.

### Domain Background
_(approx. 1-2 paragraphs)_

In this section, provide brief details on the background information of the domain from which the project is proposed. Historical information relevant to the project should be included. It should be clear how or why a problem in the domain can or should be solved. Related academic research should be appropriately cited in this section, including why that research is relevant. Additionally, a discussion of your personal motivation for investigating a particular problem in the domain is encouraged but not required.

Facial recognition is a a problem in the context of image processing that has been dealt with since over 50 years. Pioneers in this field include Woody Bledsoe, Helen Chan Wolf, and Charles Bisson.

There are competitions meassuring performance of state of the art facaial recognition algorithms like Face Recognition Grand Challenge in 2006 that evaluated performans based on a dataset showing faces with different facial expressions as wells as different lightning conditions and environments.

Applications of Facial recognition algorithms are mostly in the domain of security and defence. They include automated passport control at airports, providing a possibility of logging into an account and accessing sensetive data as well as surveillance among others.

### Problem Statement
_(approx. 1 paragraph)_

In this section, clearly describe the problem that is to be solved. The problem described should be well defined and should have at least one relevant potential solution. Additionally, describe the problem thoroughly such that it is clear that the problem is quantifiable (the problem can be expressed in mathematical or logical terms) , measurable (the problem can be measured by some metric and clearly observed), and replicable (the problem can be reproduced and occurs more than once).

The problem to be solved is to locate and recognize faces in any given images. The localization problem will be solved by standard openCV functionality (Haar Feature-based Cascade Classifier for Object Detection) and won't be benchmarked. Any localized face should be successfully classified by the face recognition algorithm. The results are compared to another state of the art face recognition algorithm (which?).

### Datasets and Inputs
_(approx. 2-3 paragraphs)_

In this section, the dataset(s) and/or input(s) being considered for the project should be thoroughly described, such as how they relate to the problem and why they should be used. Information such as how the dataset or input is (was) obtained, and the characteristics of the dataset or input, should be included with relevant references and citations as necessary It should be clear how the dataset(s) or input(s) will be used in the project and whether their use is appropriate given the context of the problem.

The problem's solution has numerous inputs:

 - VGG Face Dataset [http://www.robots.ox.ac.uk/~vgg/data/vgg_face/] [http://www.robots.ox.ac.uk/%7Evgg/publications/2015/Parkhi15/parkhi15.pdf]: The dataset consists of 2622000 images showing celebreties. It contains 2622 identities with 1000 images for each identity. The dataset is used to train CNN's that are based on ImageNet competition winning architectures (VGG-16, ResNet-50)
 
 - Pretrained VGG Face model [https://github.com/rcmalli/keras-vggface]: The VGG-16 CNN is used as a baseline. A transfer learning approach is applied to it in order to extract the CNN's feature that face recognition is applied upon. The features are further process by proprietary face recognition algorithm.

 - Haar Feature-based Cascade Classifier for Object Detection (Faces) [https://www.docs.opencv.org/2.4/modules/objdetect/doc/cascade_classification.html]: Used for face localization in order to extract the input to the CNN

### Solution Statement
_(approx. 1 paragraph)_

In this section, clearly describe a solution to the problem. The solution should be applicable to the project domain and appropriate for the dataset(s) or input(s) given. Additionally, describe the solution thoroughly such that it is clear that the solution is quantifiable (the solution can be expressed in mathematical or logical terms) , measurable (the solution can be measured by some metric and clearly observed), and replicable (the solution can be reproduced and occurs more than once).

The solution to the problem is an algorithm that detects faces in given images and matches them to faces contained in a database. Imgaes are input to the algorithm through a connected camera (webcam). The algorithm itsself can be subdivided into the following steps:

 1) Face extraction: Any shown faces on the input image are localized (by OpenCV built in face detection), extracted and scaled to a 224x224 jpg.
 2) Extraction of facial features: All extracted facial images are fed into pretrained VGG-16 CNN. A feature vector is generated.
 3) Matching feature vectors: The extracted features are compared to any feature vector within a local database. Comparism takes place based on eucledian distance. If a match is detected the respective name is shown.

### Benchmark Model
_(approximately 1-2 paragraphs)_

In this section, provide the details for a benchmark model or result that relates to the domain, problem statement, and intended solution. Ideally, the benchmark model or result contextualizes existing methods or known information in the domain and problem given, which could then be objectively compared to the solution. Describe how the benchmark model or result is measurable (can be measured by some metric and clearly observed) with thorough detail.

A Benchmark model is Pretrained VGG Face model [https://github.com/rcmalli/keras-vggface]. Real Time comparism of recorded faces to database content should perform as well as the pretrained CNN. The only difference is that VGG Face uses a pretrained classifier (consisting of fully-connected layers) whereas the proposed algorithm simply comapred facial featrues based on the eucledian distance. The comparisim will only work when comparing people that the CNN is trained upon (thus celebreties contained in the VGG Face database).

### Evaluation Metrics
_(approx. 1-2 paragraphs)_

In this section, propose at least one evaluation metric that can be used to quantify the performance of both the benchmark model and the solution model. The evaluation metric(s) you propose should be appropriate given the context of the data, the problem statement, and the intended solution. Describe how the evaluation metric(s) are derived and provide an example of their mathematical representations (if applicable). Complex evaluation metrics should be clearly defined and quantifiable (can be expressed in mathematical or logical terms).

Two classification algorithms are compared. Thus the output of the softmax layer of the classifier can be compared to calculated eucledian distance that is fed into a softmax. (does this make sense?)

### Project Design
_(approx. 1 page)_

In this final section, summarize a theoretical workflow for approaching a solution given the problem. Provide thorough discussion for what strategies you may consider employing, what analysis of the data might be required before being used, or which algorithms will be considered for your implementation. The workflow and discussion that you provide should align with the qualities of the previous sections. Additionally, you are encouraged to include small visualizations, pseudocode, or diagrams to aid in describing the project design, but it is not required. The discussion should clearly outline your intended workflow of the capstone project.

The following steps describe how the problem may be approched:

The problem is solved in Python.

1) Analysis of external dependencies:
   - Haar Feature-based Cascade Classifier for Object Detection (Faces): The algorithm will take (unnormalzed) images of abritary size and is straight forward to use.
   - VGG Face model: The algorithm takes normalized RGB-images of size 224x244. Normalized tensors of size nx244x244x3 whereas n is the batch size that the feature prediction takes place upon.
   - Required machinery: A webcam driver is required in order to read in the images. The Python OpenCV library solves this problem: cv2.VideoCapture()
2) Algorithm design: 
   - Extract faces from camera image
   - Extract facial features using CNN by making predictions upon faces
   - Compare feature vector to feature vectors in local database. Comparism is done based on eucledian distance. If distance is below threashold, a match is detected
   - On user input: record feature vector of new face and add it to the database
   
   Any of the above steps takes place in real time.

-----------

**Before submitting your proposal, ask yourself. . .**

- Does the proposal you have written follow a well-organized structure similar to that of the project template?
- Is each section (particularly **Solution Statement** and **Project Design**) written in a clear, concise and specific fashion? Are there any ambiguous terms or phrases that need clarification?
- Would the intended audience of your project be able to understand your proposal?
- Have you properly proofread your proposal to assure there are minimal grammatical and spelling mistakes?
- Are all the resources used for this project correctly cited and referenced?
