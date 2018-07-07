
import sklearn.preprocessing as prep
import tensorflow as tf

def get_data(path, data_type):
    source_train_data_filename = path+"/source_train_data"
    target_train_data_filename = path+"/target_train_data"
    source_test_data_filename = path+"/source_test_data"
    target_test_data_filename = path+"/source_test_data"
    
    source_train_data = read(source_train_data_filename)
    target_train_data = read(target_train_data_filename)
    if exists(source_test_data_filename):
        source_test_data = read(source_test_data_filename)
        source_train_data, source_test_data= standard_scale(source_train_data, source_test_data)
    else:   
        source_train_data = standard_scale(source_train_data)
        source_test_data = source_train_data
    if exists(source_test_data_filename):    
        target_test_data = read(target_test_data_filename)
        target_train_data, target_test_data= standard_scale(source_train_data, source_test_data)
    else:
        target_test_data = target_train_data
        target_train_data = source_train_data
    # do log transformation for cytof data    
    if data_type == 'cytof':
        source_train_data = preProcessCytofData(source_train_data)
        source_test_data = preProcessCytofData(source_test_data)
        target_train_data = preProcessCytofData(target_train_data)
        target_test_data = preProcessCytofData(target_test_data)
        
    return  source_train_data, target_train_data, source_test_data, target_test_data

def get_models(model_name):
    return getattr(models, model_name)()

def gradient_penalty(real, fake, f):
        def interpolate(a, b):
            shape = tf.concat((tf.shape(a)[0:1], tf.tile([1], [a.shape.ndims - 1])), axis=0)
            alpha = tf.random_uniform(shape=shape, minval=0., maxval=1.)
            inter = a + alpha * (b - a)
            inter.set_shape(a.get_shape().as_list())
            return inter

        x = interpolate(real, fake)
        pred = f(x)
        gradients = tf.gradients(pred, x)[0]
        slopes = tf.sqrt(tf.reduce_sum(tf.square(gradients), reduction_indices=range(1, x.shape.ndims)))
        gp = tf.reduce_mean((slopes - 1.)**2)
        return gp

def trainable_variables(filters=None, combine_type='or'):
    t_var = tf.trainable_variables()
    if filters is None:
        return t_var
    else:
        return tensors_filter(t_var, filters, combine_type)
    
def standard_scale(X_train, X_test):
    preprocessor = prep.StandardScaler().fit(X_train)
    X_train = preprocessor.transform(X_train)
    if X_test is not None:
        X_test = preprocessor.transform(X_test)
    return X_train, X_test    

def preProcessCytofData(data):
    return np.log(1+data)
    