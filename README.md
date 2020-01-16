# keras_warmstarter

Ever wanted to use weights of your previous model to initialize you new model? But here's the catch: you can't do this 
if the layer's shape has changed. ``keras_warmstarter`` reshape the kernel such that the new layer compute a function
which is close to the old layer. 