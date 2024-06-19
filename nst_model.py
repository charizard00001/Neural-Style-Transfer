import tensorflow as tf
import numpy as np
from PIL import Image

def tensor_to_image(tensor):
    tensor = tensor * 255
    tensor = np.array(tensor, dtype=np.uint8)
    if np.ndim(tensor) > 3:
        assert tensor.shape[0] == 1
        tensor = tensor[0]
    return Image.fromarray(tensor)

def load_img(image_path, max_dim=512):
    img = Image.open(image_path)
    long = max(img.size)
    scale = max_dim / long
    new_width = round(img.size[0] * scale)
    new_height = round(img.size[1] * scale)
    img = img.resize((new_width, new_height), Image.LANCZOS)
    img = np.array(img)

    if len(img.shape) == 2:
        img = np.stack((img,) * 3, axis=-1)
    elif img.shape[2] == 4:
        img = img[..., :3]

    img = np.expand_dims(img, axis=0)
    return img


def load_preprocess_img(image_path):
    img = load_img(image_path)
    img = tf.keras.applications.vgg19.preprocess_input(img)
    return img

def deprocess(img):
    x = img.copy()
    if len(x.shape) == 4:
        x = np.squeeze(x, 0)
    assert len(x.shape) == 3, "Input dimension must be 4 or 3"

    x[:, :, 0] += 103.939
    x[:, :, 1] += 116.779
    x[:, :, 2] += 123.68
    x = x[:, :, ::-1]

    x = np.clip(x, 0, 255).astype('uint8')
    return x

vgg = tf.keras.applications.VGG19(include_top=False, weights='imagenet')
content_layers = ['block5_conv2']
style_layers = ['block1_conv1', 'block2_conv1', 'block3_conv1', 'block4_conv1', 'block5_conv1']
num_content_layers = len(content_layers)
num_style_layers = len(style_layers)

def vgg_layers(layer_names):
    vgg = tf.keras.applications.VGG19(include_top=False, weights='imagenet')
    vgg.trainable = False
    outputs = [vgg.get_layer(name).output for name in layer_names]
    model = tf.keras.Model([vgg.input], outputs)
    return model

def get_content_loss(base_content, target):
    return tf.reduce_mean(tf.square(base_content - target))

def gram_matrix(input_tensor):
    input_shape = tf.shape(input_tensor)
    reshaped_tensor = tf.reshape(input_tensor, [input_shape[1] * input_shape[2], input_shape[3]])
    gram = tf.matmul(reshaped_tensor, reshaped_tensor, transpose_a=True)
    num_locations = tf.cast(input_shape[1] * input_shape[2], tf.float32)
    return gram / num_locations

def get_style_loss(base_style, gram_target):
    gram_style = gram_matrix(base_style)
    return tf.reduce_mean(tf.square(gram_style - gram_target))

def get_feature_representations(model, content_path, style_path):
    content_image = load_preprocess_img(content_path)
    style_image = load_preprocess_img(style_path)

    style_outputs = model(style_image)
    content_outputs = model(content_image)

    print("Style Outputs Shape:", [output.shape for output in style_outputs])
    print("Content Outputs Shape:", [output.shape for output in content_outputs])

    style_features = [style_layer for style_layer in style_outputs[:num_style_layers]]
    content_features = [content_layer for content_layer in content_outputs[num_style_layers:]]

    print("Style Features Shape:", [feature.shape for feature in style_features])
    print("Content Features Shape:", [feature.shape for feature in content_features])

    return style_features, content_features

def compute_loss(model, loss_weights, init_image, gram_style_features, content_features):
    style_weight, content_weight = loss_weights

    model_outputs = model(init_image)

    style_output_features = model_outputs[:num_style_layers]
    content_output_features = model_outputs[num_style_layers:]

    print("Model Output Style Features Shape:", [feature.shape for feature in style_output_features])
    print("Model Output Content Features Shape:", [feature.shape for feature in content_output_features])

    style_score = 0
    content_score = 0

    weight_per_style_layer = 1.0 / float(num_style_layers)
    for target_style, comb_style in zip(gram_style_features, style_output_features):
        style_score += weight_per_style_layer * get_style_loss(comb_style, target_style)

    weight_per_content_layer = 1.0 / float(num_content_layers)
    for target_content, comb_content in zip(content_features, content_output_features):
        content_score += weight_per_content_layer * get_content_loss(comb_content, target_content)

    style_score *= style_weight
    content_score *= content_weight

    loss = style_score + content_score
    return loss, style_score, content_score

def compute_grads(cfg):
    with tf.GradientTape() as tape: 
        all_loss = compute_loss(**cfg)
    total_loss = all_loss[0]
    return tape.gradient(total_loss, cfg['init_image']), all_loss

def run_style_transfer(content_path, style_path, num_iterations=100, content_weight=1e3, style_weight=1e-2): 
    model = vgg_layers(style_layers + content_layers)
    for layer in model.layers:
        layer.trainable = False

    style_features, content_features = get_feature_representations(model, content_path, style_path)
    gram_style_features = [gram_matrix(style_feature) for style_feature in style_features]

    init_image = load_preprocess_img(content_path)
    init_image = tf.Variable(init_image, dtype=tf.float32)

    opt = tf.optimizers.Adam(learning_rate=5, beta_1=0.99, epsilon=1e-1)

    best_loss, best_img = float('inf'), None

    loss_weights = (style_weight, content_weight)
    cfg = {
        'model': model,
        'loss_weights': loss_weights,
        'init_image': init_image,
        'gram_style_features': gram_style_features,
        'content_features': content_features
    }

    norm_means = np.array([103.939, 116.779, 123.68])
    min_vals = -norm_means
    max_vals = 255 - norm_means   

    for i in range(num_iterations):
        grads, all_loss = compute_grads(cfg)
        loss, style_score, content_score = all_loss
        opt.apply_gradients([(grads, init_image)])
        clipped = tf.clip_by_value(init_image, min_vals, max_vals)
        init_image.assign(clipped)

        if loss < best_loss:
            best_loss = loss
            best_img = init_image.numpy()

        if i % 100 == 0:
            print('Iteration: {}'.format(i))
            print('Total loss: {:.4e}, style loss: {:.4e}, content loss: {:.4e}'.format(loss, style_score, content_score))

    return best_img, best_loss
