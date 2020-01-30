from tqdm import tqdm
import numpy as np
import pickle

class SynthDataGenerator():
  def __init__(self, model=None, n_classes=1, n_samples=1, dist_thresh=0, noise_variance=0.2):
    self.model = model
    self.n_classes = n_classes
    self.n_samples = n_samples
    self.noise_variance = noise_variance
    self.dist_thresh = dist_thresh

    self.X = []
    self.y = []
    self.already_gen_noise = []
    self.latent_dims = model.layers[0].output_shape[1]

  def generate(self):
    print("\ngenerating the dataset...")
    for i in tqdm(range(self.n_classes)):
      target_noise = self.generate_vect_class()
      self.X.append(self.from_latent_to_image(target_noise))
      self.y.append(i)

      for j in range(self.n_samples-1):
        self.X.append(self.from_latent_to_image(
            self.generate_vect_sample(target_noise)
        ))    
        self.y.append(i)
    print("\ndataset generated")

  # generate a valid noise vector
  def generate_vect_class(self):
    noise = np.random.normal(0, 1, (1,100))
    is_valid_noise = np.all([self.dist(noise, n) > self.dist_thresh for n in self.already_gen_noise])

    if is_valid_noise: 
      self.already_gen_noise.append(noise)
      return noise

    return self.generate_vect_class()
    

  def generate_vect_sample(self, target_noise):
    return target_noise +  np.random.normal(0, self.noise_variance, self.latent_dims)

  def from_latent_to_image(self, noise):
    return self.preprocess(self.model.predict(noise))

  def get_data(self):
    return self.X, self.y

  def save_data(self, pickle_path=None):

    assert (pickle_path.endswith('.pickle')), "The file must have the '.pickle' extension"

    try:
      _file = open(pickle_path, 'wb')    
      
      pickle.dump(((np.array(self.X), np.array(self.y))), _file)
      _file.close()
      print("File saved successfully")
    except Exception as e:
      print("The file is not saved, reasons: " + e)

  @staticmethod
  def preprocess(im):
    im = (im+1)*127.5
    return np.squeeze(im)

  @staticmethod
  def dist(a, b):
    return np.linalg.norm(a - b)
