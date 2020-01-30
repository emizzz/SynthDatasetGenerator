# SynthDatasetGenerator
A (Gan based) dataset generator 

```
# load the gan model (pretrained)
val_model = load_model(project_path + "your_pretrained_keras_gan_model.h5")

# initialize the generator
synth_data_generator = SynthDataGenerator(
    model = val_model,
    n_classes = 500,
    n_samples = 100,
    dist_thresh = 10,     
    noise_variance=0.30
)

# start the generation process
synth_data_generator.generate()

# get the data (data, labels)
X, y = synth_data_generator.get_data()

# save data
synth_data_generator.save_data(project_path + "/synth_data_gan.pickle")
```
