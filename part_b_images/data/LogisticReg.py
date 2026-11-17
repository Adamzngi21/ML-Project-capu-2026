import numpy as np
import tensorflow as tf
import tensorflow_datasets as tfds


dataset, info = tfds.load('tf_flowers', with_info=True, as_supervised=True)
full_dataset = dataset['train']
class_names = info.features['label'].names  

IMG_SIZE = 128  

def resize_and_normalize(image, label):
    image = tf.image.resize(image, (IMG_SIZE, IMG_SIZE))
    image = tf.cast(image, tf.float32) / 255.0
    return image, label

full_dataset = full_dataset.map(resize_and_normalize)



def rgb_to_hsv_np(img):
    r, g, b = img[:, :, 0], img[:, :, 1], img[:, :, 2]
    maxc = np.max(img, axis=2)
    minc = np.min(img, axis=2)
    v = maxc
    deltac = maxc - minc
    s = np.where(maxc != 0, deltac / (maxc + 1e-8), 0.0)

    rc = (maxc - r) / (deltac + 1e-8)
    gc = (maxc - g) / (deltac + 1e-8)
    bc = (maxc - b) / (deltac + 1e-8)

    h = np.zeros_like(maxc)
    h = np.where((maxc == r), bc - gc, h)
    h = np.where((maxc == g), 2.0 + rc - bc, h)
    h = np.where((maxc == b), 4.0 + gc - rc, h)
    h = (h / 6.0) % 1.0
    h = np.where(deltac == 0, 0.0, h)
    return np.stack([h, s, v], axis=2)


def get_hsv_hist(hsv_img, bins=32):
    h_hist, _ = np.histogram(hsv_img[:, :, 0], bins=bins, range=(0, 1))
    s_hist, _ = np.histogram(hsv_img[:, :, 1], bins=bins, range=(0, 1))
    v_hist, _ = np.histogram(hsv_img[:, :, 2], bins=bins, range=(0, 1))
    hist = np.concatenate([h_hist, s_hist, v_hist])
    return hist / (np.sum(hist) + 1e-8)


def extract_features(img, bins=32):
    hsv_full = rgb_to_hsv_np(img)

    full_color_feat = get_hsv_hist(hsv_full, bins=bins)

    margin = IMG_SIZE // 4  
    hsv_center = hsv_full[margin:-margin, margin:-margin, :]
    center_color_feat = get_hsv_hist(hsv_center, bins=bins)

    gray = np.mean(img, axis=2)
    gx = np.diff(gray, axis=1)[:-1, :]  
    gy = np.diff(gray, axis=0)[:, :-1]  
    magnitude = np.sqrt(gx**2 + gy**2)
    orientation = np.arctan2(gy, gx)

    edge_hist, _ = np.histogram(orientation, bins=bins, range=(-np.pi, np.pi), weights=magnitude)
    edge_feat = edge_hist / (np.sum(edge_hist) + 1e-8)

    return np.concatenate([full_color_feat, center_color_feat, edge_feat])


print("Featruue Extracting...")
X_list, y_list = [], []
for img, label in tfds.as_numpy(full_dataset):
    X_list.append(extract_features(img, bins=32))
    y_list.append(label)

X = np.array(X_list) 
y = np.array(y_list)  



indices = np.arange(X.shape[0])
np.random.seed(42)
np.random.shuffle(indices)

split_idx = int(0.8 * len(indices))
train_idx, test_idx = indices[:split_idx], indices[split_idx:]

X_train, y_train = X[train_idx], y[train_idx]
X_test, y_test = X[test_idx], y[test_idx]

mean = np.mean(X_train, axis=0, keepdims=True)
std = np.std(X_train, axis=0, keepdims=True) + 1e-8

X_train = (X_train - mean) / std
X_test = (X_test - mean) / std



class MultinomialLogisticRegression:
    def __init__(self, learning_rate=0.05, epochs=800, batch_size=64, lambda_reg=0.01):
        self.lr = learning_rate
        self.epochs = epochs
        self.batch_size = batch_size
        self.lambda_reg = lambda_reg  # معامل L2 Regularization لمنع الـ Overfitting
        self.W = None
        self.b = None
        self.loss_history = []

    def _one_hot(self, y, num_classes):
        one_hot_y = np.zeros((y.shape[0], num_classes))
        one_hot_y[np.arange(y.shape[0]), y] = 1
        return one_hot_y

    def _softmax(self, z):
        exp_z = np.exp(z - np.max(z, axis=1, keepdims=True))
        return exp_z / np.sum(exp_z, axis=1, keepdims=True)

    def _compute_loss(self, y_true_one_hot, y_pred_probs):
        epsilon = 1e-15
        y_pred_probs = np.clip(y_pred_probs, epsilon, 1 - epsilon)
        cross_entropy = -np.mean(np.sum(y_true_one_hot * np.log(y_pred_probs), axis=1))
        l2_penalty = 0.5 * self.lambda_reg * np.sum(self.W ** 2)
        return cross_entropy + l2_penalty

    def fit(self, X, y):
        num_samples, num_features = X.shape
        num_classes = len(np.unique(y))

        np.random.seed(42)
        self.W = np.random.randn(num_features, num_classes) * 0.01
        self.b = np.zeros((1, num_classes))

        y_one_hot = self._one_hot(y, num_classes)

        for epoch in range(self.epochs):
            indices = np.arange(num_samples)
            np.random.shuffle(indices)
            X_shuffled = X[indices]
            y_shuffled = y_one_hot[indices]

            for i in range(0, num_samples, self.batch_size):
                X_batch = X_shuffled[i:i + self.batch_size]
                y_batch = y_shuffled[i:i + self.batch_size]
                m = X_batch.shape[0]

                Z = np.dot(X_batch, self.W) + self.b
                Y_hat = self._softmax(Z)

                dZ = Y_hat - y_batch
                dW = (1 / m) * np.dot(X_batch.T, dZ) + (self.lambda_reg * self.W)
                db = (1 / m) * np.sum(dZ, axis=0, keepdims=True)

                
                self.W -= self.lr * dW
                self.b -= self.lr * db

            if (epoch + 1) % 100 == 0 or epoch == 0:
                full_probs = self.predict_proba(X)
                loss = self._compute_loss(y_one_hot, full_probs)
                self.loss_history.append(loss)
                train_acc = self.score(X, y)
                print(f"Epoch {epoch+1:3d}/{self.epochs} | Loss: {loss:.4f} | Train Acc: {train_acc*100:.2f}%")

    def predict_proba(self, X):
        Z = np.dot(X, self.W) + self.b
        return self._softmax(Z)

    def predict(self, X):
        return np.argmax(self.predict_proba(X), axis=1)

    def score(self, X, y):
        return np.mean(self.predict(X) == y)



print(f"\n Matrix  {X_train.shape}")
model = MultinomialLogisticRegression(learning_rate=0.05, epochs=800, batch_size=64, lambda_reg=0.01)
model.fit(X_train, y_train)

train_accuracy = model.score(X_train, y_train)
test_accuracy = model.score(X_test, y_test)

print("\n" + "="*45)
print(f" Train Accuracy: {train_accuracy * 100:.2f}%")
print(f" Test Accuracy : {test_accuracy * 100:.2f}%")
print("="*45)


y_pred = model.predict(X_test)
print("\n accuracy detailds for each class:")
for idx, name in enumerate(class_names):
    class_mask = (y_test == idx)
    class_acc = np.mean(y_pred[class_mask] == y_test[class_mask])
    print(f"- {name:12s}: {class_acc * 100:.2f}%")