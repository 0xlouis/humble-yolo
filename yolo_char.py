import keras
from keras.models import Sequential, Model
from keras.layers import Dense, Dropout, Flatten, Input, Reshape
from keras.layers import Conv2D, MaxPooling2D, BatchNormalization
from keras.layers.advanced_activations import LeakyReLU, PReLU, ELU
from keras.preprocessing.image import ImageDataGenerator, array_to_img, img_to_array, load_img
from keras import backend as K
from keras.models import load_model
from keras.utils import np_utils

import numpy as np

from PIL import Image, ImageDraw, ImageFont

import matplotlib.pyplot as plt

nb_boxes = 1#lossYOLO_MultiDimensional n'est pas implementée pour nb_boxes > 1 ; Utiliser lossYOLO_1D a la place (TODO)
grid_w   = 4
grid_h   = 4
cell_w   = 16
cell_h   = 16
img_w    = grid_w * cell_w
img_h    = grid_h * cell_h

VOCAB = ['A', 'B', 'C']
L_VOC = len(VOCAB)

FONTS=['datas/fonts/0.ttf',
       'datas/fonts/1.ttf',
       'datas/fonts/2.otf',
       'datas/fonts/3.ttf',
       'datas/fonts/4.otf',
       'datas/fonts/5.otf',
       'datas/fonts/6.otf']

BatchSize = 4

def tokenizeY(strTab, vocab):
    #Texte ver OneHot
    onehotData = []
    for char in strTab:
        onehotData.append(vocab.index(char))
    
    data    = np.array(onehotData)
    encoded = np_utils.to_categorical(data, num_classes=len(vocab))
    
    return encoded

def genElement(p=0.1, dbg=None):
    while True:
        #Genere l'image
        img = Image.new('RGB', (img_w, img_h))
        drw = ImageDraw.Draw(img)
        
        #Liste des predictions a faire (batch)
        prd = []
        
        #Utilise une generation deterministe (utile a des fins de debug)
        #Shape : (grid_h * grid_w * nb_box, [classe, x, y, largeur, hauteur, confidence])
        if dbg != None:
            i = 0
            for y in range(grid_h):
                for x in range(grid_w):
                    row = dbg[i]
                    i  += 1
                    
                    #Choisir le caractere
                    carac = row[0]
                    
                    #Contient les elements de prediction de la cellule courrante
                    p_elm = []
                    p_elm.extend(tokenizeY(carac, VOCAB).tolist()[0])
                    
                    #Pour chaque box
                    for z in range(nb_boxes):
                        #Determine la forme du caractere
                        c_pol = 25
                        c_fnt = ImageFont.truetype(FONTS[0], c_pol)
                        c_rgb = (np.random.randint(64, 256), np.random.randint(64, 256), np.random.randint(64, 256))
                        c_wdt, c_hgt = c_fnt.getsize(carac)
                        
                        #Choisi une coordonnée dans la region (le centre de la boite doit etre inclu dans la region)
                        x_box = row[z*(2+2+1) + 1]
                        y_box = row[z*(2+2+1) + 2]
                        
                        #Determine la position absolue du caractere
                        x_car = (cell_w * x) + x_box - (c_wdt / 2.0)
                        y_car = (cell_h * y) + y_box - (c_hgt / 2.0)
                        
                        #Dessine le carctere
                        drw.text((x_car, y_car), carac, font=c_fnt, fill=c_rgb)
                        
                        #Peuple l'element de prediction
                        p_elm.append(x_box / cell_w)
                        p_elm.append(y_box / cell_h)
                        p_elm.append(c_wdt /  img_w)
                        p_elm.append(c_hgt /  img_h)
                        p_elm.append(1.0)

                    prd.append(p_elm)
            yield img, prd
            continue

        #Parcour de l'image dans le sens de lecture
        for y in range(grid_h):
            for x in range(grid_w):
                #Determine si la grille va contenir un caractere
                c_prb = p
                c_win = np.random.choice([True, False], p=[c_prb, 1.0-c_prb])
                
                #Si la grille doit contenir un caractere
                if c_win:
                    #Determine le nombres de caracteres de meme type que la grille doit contenir
                    c_cpt = np.random.randint(nb_boxes)
                    
                    #Choisir un caractere au hasard
                    carac = np.random.choice(VOCAB)
                    
                    #Contient les elements de prediction de la cellule courrante
                    p_elm = []
                    p_elm.extend(tokenizeY(carac, VOCAB).tolist()[0])
                    
                    #Pour chaque box
                    for z in range(nb_boxes):
                        #Place un caractere si le compteur n'est pas epuisé
                        if z <= c_cpt:
                            #Determine la forme du caractere
                            c_pol = np.random.randint(15, 35)
                            c_fnt = ImageFont.truetype(np.random.choice(FONTS), c_pol)
                            c_rgb = (np.random.randint(64, 256), np.random.randint(64, 256), np.random.randint(64, 256))
                            c_wdt, c_hgt = c_fnt.getsize(carac)
                            
                            #Choisi une coordonnée dans la region (le centre de la boite doit etre inclu dans la region)
                            x_box = np.random.randint(0, cell_w)
                            y_box = np.random.randint(0, cell_h)
                            
                            #Determine la position absolue du caractere
                            x_car = (cell_w * x) + x_box - (c_wdt / 2.0)
                            y_car = (cell_h * y) + y_box - (c_hgt / 2.0)
                            
                            #Dessine le carctere
                            drw.text((x_car, y_car), carac, font=c_fnt, fill=c_rgb)
                            
                            #Peuple l'element de prediction
                            p_elm.append(x_box / cell_w)
                            p_elm.append(y_box / cell_h)
                            p_elm.append(c_wdt /  img_w)
                            p_elm.append(c_hgt /  img_h)
                            p_elm.append(1.0)
                        else:
                            #Peuple l'element de prediction
                            p_elm.append(0.0)
                            p_elm.append(0.0)
                            p_elm.append(0.0)
                            p_elm.append(0.0)
                            p_elm.append(0.0)
                else:
                    #Contient les elements de prediction de la cellule courrante
                    p_elm = []
                    p_elm.extend([0.0] * L_VOC)
                    
                    #Pour chaque box
                    for z in range(nb_boxes):
                        #Peuple l'element de prediction
                        p_elm.append(0.0)
                        p_elm.append(0.0)
                        p_elm.append(0.0)
                        p_elm.append(0.0)
                        p_elm.append(0.0)

                prd.append(p_elm)

        yield img, prd

def genDataset(N=1, dbg=None):
    gen = genElement(p=0.15, dbg=dbg)
    
    while True:
        n_train = N
        x_train = []
        y_train = []
        
        for i in range(n_train):
            x, y = next(gen)
            x_train.append(img_to_array(x))
            y_train.append(y)
        
        yield np.array(x_train), np.array(y_train)

def getBoxedImg(img, prd, seuil=0.25):
    drw = ImageDraw.Draw(img)
    out = np.reshape(prd, (grid_h, grid_w, L_VOC+nb_boxes*(2+2+1)))
    
    #Parcour de l'image dans le sens de lecture
    for y in range(grid_h):
        for x in range(grid_w):
            box = out[y, x]                
            b_c = VOCAB[np.argmax(box[0:L_VOC])]
            
            for z in range(nb_boxes):
                b_f = box[L_VOC + z*(2+2+1) + 4]
                
                if b_f < seuil:
                    continue
                
                b_w = box[L_VOC + z*(2+2+1) + 2] * img_w
                b_h = box[L_VOC + z*(2+2+1) + 3] * img_h
                b_x = box[L_VOC + z*(2+2+1) + 0] * cell_w + (cell_w * x) - (b_w / 2.0)
                b_y = box[L_VOC + z*(2+2+1) + 1] * cell_h + (cell_h * y) - (b_h / 2.0)
                
                #drw.text((b_x, b_y), b_c)
                drw.rectangle((b_x, b_y, b_x+b_w, b_y+b_h), outline=(255, 0, 0))
                print('({0},{1},{2}) : \'{3}\' > {4:.3f}'.format(y, x, z, b_c, b_f))
    print('')
    
    return img

def lossYOLO_MultiDimensional(y_true, y_pred):
    #Constantes d'apprentissages
    lambda_coord = 5
    lambda_noobj = 0.5
    n_features   = L_VOC+nb_boxes*(2+2+1)
    Shape_Full   = (BatchSize, grid_h*grid_w, n_features)
    Shape_Class  = (BatchSize, grid_h*grid_w, L_VOC)
    Shape_XY     = (BatchSize, grid_h*grid_w, 1)
    Shape_WH     = (BatchSize, grid_h*grid_w, 1)
    Shape_CF     = (BatchSize, grid_h*grid_w, 1)
    
    np_mask_x  = np.zeros(Shape_Full)
    np_mask_x[...,L_VOC+0] = 1
    Mask_X     = K.variable(np_mask_x)
    
    np_mask_y  = np.zeros(Shape_Full)
    np_mask_y[...,L_VOC+1] = 1
    Mask_Y     = K.variable(np_mask_y)
    
    Mask_XY    = Mask_X + Mask_Y
    
    np_mask_w  = np.zeros(Shape_Full)
    np_mask_w[...,L_VOC+2] = 1
    Mask_W     = K.variable(np_mask_w)
    
    np_mask_h  = np.zeros(Shape_Full)
    np_mask_h[...,L_VOC+3] = 1
    Mask_H     = K.variable(np_mask_h)
    
    Mask_WH    = Mask_W + Mask_H
    
    np_mask_cl = np.zeros(Shape_Full)
    np_mask_cl[...,:L_VOC] = 1
    Mask_CL    = K.variable(np_mask_cl)
    
    np_mask_cf = np.zeros(Shape_Full)
    np_mask_cf[...,L_VOC+4] = 1
    Mask_CF    = K.variable(np_mask_cf)
    
    #Variables
    loss = K.zeros(Shape_Full)

    #Exctraction haut niveau des variables
    y_true_class = K.reshape(y_true[..., :L_VOC], (-1, L_VOC))
    y_pred_class = K.reshape(y_pred[..., :L_VOC], (-1, L_VOC))
    y_true_boxes = K.transpose(K.reshape(y_true[..., L_VOC:], (-1, 2+2+1)))
    y_pred_boxes = K.transpose(K.reshape(y_pred[..., L_VOC:], (-1, 2+2+1)))
    y_true_x     = y_true_boxes[0]
    y_pred_x     = y_pred_boxes[0]
    y_true_y     = y_true_boxes[1]
    y_pred_y     = y_pred_boxes[1]
    y_true_dw    = y_true_boxes[2]
    y_pred_dw    = y_pred_boxes[2]
    y_true_dh    = y_true_boxes[3]
    y_pred_dh    = y_pred_boxes[3]
    y_true_cf    = y_true_boxes[4]
    y_pred_cf    = y_pred_boxes[4]

    ### Loss des classes
    cl_errors    = K.square(y_true_class - y_pred_class)
    cl_errors    = K.repeat_elements(cl_errors, nb_boxes, 0)
    
    pocl_mask    = K.repeat_elements(y_true_cf, L_VOC, 0)
    pocl_loss    = K.flatten(cl_errors) * pocl_mask
    pocl_fill    = K.zeros((BatchSize, grid_h*grid_w, n_features-L_VOC))
    pocl_loss    = K.reshape(pocl_loss, (Shape_Class))
    pocl_loss    = K.concatenate([pocl_loss, pocl_fill])

    nocl_mask    = K.repeat_elements(K.abs(1.0 - y_true_cf), L_VOC, 0)
    nocl_loss    = K.flatten(cl_errors) * nocl_mask
    nocl_fill    = K.zeros((BatchSize, grid_h*grid_w, n_features-L_VOC))
    nocl_loss    = K.reshape(nocl_loss, (Shape_Class))
    nocl_loss    = K.concatenate([nocl_loss, nocl_fill])

    ### Loss des positions relatives X et Y
    xy_loss      = (K.square(y_true_x - y_pred_x) + K.square(y_true_y - y_pred_y)) * y_true_cf
    xy_loss      = K.reshape(xy_loss, (Shape_XY))
    xy_loss      = K.repeat_elements(xy_loss, n_features, -1)
    xy_loss      = xy_loss * Mask_XY

    ### Loss des dimensions normalisées W et H
    wh_loss      = (K.square(K.sqrt(y_true_dw) - K.sqrt(y_pred_dw)) + K.square(K.sqrt(y_true_dh) - K.sqrt(y_pred_dh))) * y_true_cf
    wh_loss      = K.reshape(wh_loss, (Shape_WH))
    wh_loss      = K.repeat_elements(wh_loss, n_features, -1)
    wh_loss      = wh_loss * Mask_WH

    ### Loss des confidences
    #Converti les données en pixel (utile pour le debug)
    px_tx = y_true_x  * cell_w
    px_ty = y_true_y  * cell_h
    px_tw = y_true_dw * img_w
    px_th = y_true_dh * img_h
    px_px = y_pred_x  * cell_w
    px_py = y_pred_y  * cell_h
    px_pw = y_pred_dw * img_w
    px_ph = y_pred_dh * img_h
    
    #Calcul des intersections (xw) des longueurs
    aw = px_tw
    bw = px_pw
    iw = K.abs(px_tx - px_px)
    jw = K.abs((px_tx + px_tw) - (px_px + px_pw))
    xw = K.maximum(K.zeros_like(aw), (aw + bw - iw - jw) / 2.0)
    
    #Calcul des intersections (xh) des hauteurs
    ah = px_ty
    bh = px_ph
    ih = K.abs(px_ty - px_py)
    jh = K.abs((px_ty + px_th) - (px_py + px_ph))
    xh = K.maximum(K.zeros_like(ah), (ah + bh - ih - jh) / 2.0)

    #Calcul des surfaces d'intersections et d'unions
    intx_area = xw * xh
    true_area = px_tw * px_th
    pred_area = px_pw * px_ph
    unio_area = pred_area + true_area - intx_area

    #Calcul du IOU
    iou = intx_area / unio_area

    #Calcul du loss de la confidence    
    cf_loss   = K.square(y_true_cf * iou - y_pred_cf)
    cf_loss   = K.reshape(cf_loss, (Shape_CF))
    cf_loss   = K.repeat_elements(cf_loss, n_features, -1)
    cf_loss   = cf_loss * Mask_CF

    ### Calcul du loss final
    loss = lambda_coord * (xy_loss + wh_loss) + pocl_loss + lambda_noobj * nocl_loss + cf_loss

    return loss

def lossYOLO_1D(y_true, y_pred):
    #Constantes d'apprentissages
    lambda_coord = 5
    lambda_noobj = 0.5

    #Exctraction haut niveau des variables
    y_true_class = K.reshape(y_true[..., :L_VOC], (-1, L_VOC))
    y_pred_class = K.reshape(y_pred[..., :L_VOC], (-1, L_VOC))
    y_true_boxes = K.transpose(K.reshape(y_true[..., L_VOC:], (-1, 2+2+1)))
    y_pred_boxes = K.transpose(K.reshape(y_pred[..., L_VOC:], (-1, 2+2+1)))
    y_true_x     = y_true_boxes[0]
    y_pred_x     = y_pred_boxes[0]
    y_true_y     = y_true_boxes[1]
    y_pred_y     = y_pred_boxes[1]
    y_true_dw    = y_true_boxes[2]
    y_pred_dw    = y_pred_boxes[2]
    y_true_dh    = y_true_boxes[3]
    y_pred_dh    = y_pred_boxes[3]
    y_true_cf    = y_true_boxes[4]
    y_pred_cf    = y_pred_boxes[4]

    ### Loss des classes
    cl_errors    = K.sum(K.square(y_true_class - y_pred_class), axis=-1)
    cl_errors    = K.repeat_elements(cl_errors, nb_boxes, 0)
    pocl_loss    = K.sum(cl_errors * y_true_cf)
    nocl_loss    = K.sum(cl_errors * K.abs(1.0 - y_true_cf))

    ### Loss des positions relatives X et Y
    xy_loss      = K.sum((K.square(y_true_x - y_pred_x)  + 
                          K.square(y_true_y - y_pred_y)) * y_true_cf)

    ### Loss des dimensions normalisées W et H
    wh_loss      = K.sum((K.square(K.sqrt(y_true_dw) - K.sqrt(y_pred_dw))  + 
                          K.square(K.sqrt(y_true_dh) - K.sqrt(y_pred_dh))) * y_true_cf)

    ### Loss des confidences
    #Converti les données en pixel (utile pour le debug)
    px_tx = y_true_x  * cell_w
    px_ty = y_true_y  * cell_h
    px_tw = y_true_dw * img_w
    px_th = y_true_dh * img_h
    px_px = y_pred_x  * cell_w
    px_py = y_pred_y  * cell_h
    px_pw = y_pred_dw * img_w
    px_ph = y_pred_dh * img_h
    
    #Calcul des intersections (xw) des longueurs
    aw = px_tw
    bw = px_pw
    iw = K.abs(px_tx - px_px)
    jw = K.abs((px_tx + px_tw) - (px_px + px_pw))
    xw = K.maximum(K.zeros_like(aw), (aw + bw - iw - jw) / 2.0)
    
    #Calcul des intersections (xh) des hauteurs
    ah = px_ty
    bh = px_ph
    ih = K.abs(px_ty - px_py)
    jh = K.abs((px_ty + px_th) - (px_py + px_ph))
    xh = K.maximum(K.zeros_like(ah), (ah + bh - ih - jh) / 2.0)

    #Calcul des surfaces d'intersections et d'unions
    intx_area = xw * xh
    true_area = px_tw * px_th
    pred_area = px_pw * px_ph
    unio_area = pred_area + true_area - intx_area

    #Calcul du IOU
    iou = intx_area / unio_area

    #Calcul du loss de la confidence    
    cf_loss   = K.sum(K.square(y_true_cf * iou - y_pred_cf))

    #Normalise les loss
    loss_norm  = grid_w * grid_h * nb_boxes
    pocl_loss /= loss_norm
    nocl_loss /= loss_norm
    xy_loss   /= loss_norm
    wh_loss   /= loss_norm
    cf_loss   /= loss_norm

    #Calcul du loss final
    loss = lambda_coord * (xy_loss + wh_loss) + pocl_loss + lambda_noobj * nocl_loss + cf_loss
    
    return loss

def createModel():
    #Réseau VGG like (custom)
    model = Sequential()
    model.add(Conv2D( 16, kernel_size=(3, 3), padding='same', activation='elu', data_format="channels_last", input_shape=(img_h, img_w, 3)))
    model.add(Conv2D( 16, kernel_size=(3, 3), padding='same', activation='elu', data_format="channels_last"))
    model.add(BatchNormalization())
    model.add(MaxPooling2D(pool_size=2))
    model.add(Conv2D( 32, kernel_size=(3, 3), padding='same', activation='elu', data_format="channels_last"))
    model.add(Conv2D( 32, kernel_size=(3, 3), padding='same', activation='elu', data_format="channels_last"))
    model.add(BatchNormalization())
    model.add(MaxPooling2D(pool_size=2))
    model.add(Conv2D( 64, kernel_size=(3, 3), padding='same', activation='elu', data_format="channels_last"))
    model.add(Conv2D( 64, kernel_size=(3, 3), padding='same', activation='elu', data_format="channels_last"))
    model.add(BatchNormalization())
    model.add(MaxPooling2D(pool_size=2))
    model.add(Conv2D(128, kernel_size=(3, 3), padding='same', activation='elu', data_format="channels_last"))
    model.add(Conv2D(128, kernel_size=(3, 3), padding='same', activation='elu', data_format="channels_last"))
    model.add(BatchNormalization())
    model.add(MaxPooling2D(pool_size=2))
    model.add(Flatten())
    model.add(Dense(256, activation='sigmoid'))
    model.add(Dense(grid_w * grid_h * (L_VOC + nb_boxes * (2+2+1)), activation='sigmoid'))
    model.add(Reshape((grid_w * grid_h, (L_VOC + nb_boxes * (2+2+1)))))

    optim = keras.optimizers.Adam(lr=0.0001, decay=0.00)
    model.compile(loss=lossYOLO_MultiDimensional, optimizer=optim)

    return model

# #######
# #Sample
# n_test = 4
# t_gene = genDataset(n_test)
# x_test, y_test = next(t_gene)
# #%matplotlib qt
# for i in range(n_test):
#     img = getBoxedImg(array_to_img(x_test[i]), y_test[i])
#     plt.subplot(2, 2, i+1)
#     plt.imshow(img)
# plt.show()


# ######
# #Train
# model = createModel()
# model.summary()
# # model.load_weights('weights.h5')
# model.fit_generator(genDataset(BatchSize), steps_per_epoch=256, epochs=50)
# model.save_weights('weights.h5')

#####
#Sample pred
model = createModel()
model.load_weights('weights.h5')
n_test = 5*5
t_gene = genDataset(n_test)
x_test, y_test = next(t_gene)
y = model.predict(x_test, verbose=1)
#%matplotlib qt
for i in range(n_test):
    img = getBoxedImg(array_to_img(x_test[i]), y[i], 0.1)
    plt.subplot(5, 5, i+1)
    plt.imshow(img)
plt.show()
