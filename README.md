# Gun-Detection
Real time gun detection is one of the pivotal things to improve the survelliance methods. Previosuly, infrared rays were used to detect the concealed weapons which slows down the gun detection process. I used a Tensorflow-based implementation of VGGNet network to train. I used 5 types of guns to train my network: Assault Rifle, Battle Rifle, Bullpup, Pistols and Revolver.

<h2>Preprocessing</h2>
In the preprocessing step, I scaled the images using standard deviation which was caculated from the sample data. Also, I zero sampled the images using the mean calculated from the sample data. These pre-processing methods were used both during the training and testing time.

<h2>Database</h2>
I implemented a web crawler to download relevant images from the Internet Movie Firearms Database. I used around 5200 images in total for training and validation purposes in the ratio of 4:1 respectively. The dataset can be requested by dropping me a mail at brihati1373@gmail.com with subject "Gun Firearm Database"

<h2>VGG Model</h2>
<p>network = input_data(shape = [None, constants.IMAGE_HEIGHT, constants.IMAGE_WIDTH, 3])</p>

<p>network = conv_2d(network, 64, 3, activation='relu')</p>
<p>network = conv_2d(network, 64, 3, activation='relu')</p>
<p>network = max_pool_2d(network, 2, strides=2)</p>

<p>network = conv_2d(network, 128, 3, activation='relu')</p>
<p>network = conv_2d(network, 128, 3, activation='relu')</p>
<p>network = max_pool_2d(network, 2, strides=2)</p>

<p>network = conv_2d(network, 256, 3, activation='relu')</p>
<p>network = conv_2d(network, 256, 3, activation='relu')</p>
<p>network = conv_2d(network, 256, 3, activation='relu')</p>
<p>network = max_pool_2d(network, 2, strides=2)</p>

<p>network = conv_2d(network, 512, 3, activation='relu')</p>
<p>network = conv_2d(network, 512, 3, activation='relu')</p>
<p>network = conv_2d(network, 512, 3, activation='relu')</p>
<p>network = max_pool_2d(network, 2, strides=2)</p>

<p>network = conv_2d(network, 512, 3, activation='relu')</p>
<p>network = conv_2d(network, 512, 3, activation='relu')</p>
<p>network = conv_2d(network, 512, 3, activation='relu')</p>
<p>network = max_pool_2d(network, 2, strides=2)</p>

<p>network = fully_connected(network, 4096, activation='relu')</p>
<p>network = dropout(network, 0.5)</p>
<p>network = fully_connected(network, 4096, activation='relu')</p>
<p>network = dropout(network, 0.5)</p>
<p>network = fully_connected(network, 5, activation='softmax')</p>

<p>network = regression(network, optimizer='adam',
                          loss='categorical_crossentropy',
                          learning_rate=0.0001)</p>

<h2>Results</h2>
The model achieved promising results having 80% as training accuracy and 78% validation accuracy. 
