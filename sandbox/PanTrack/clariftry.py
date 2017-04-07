from clarifai import rest
from clarifai.rest import ClarifaiApp
from clarifai.rest import Image as ClImage

app = ClarifaiApp('NLrJmndR_4yDuKWgEOulLE4POX8J_PKp3O8XvcnC', 'lz5lwrp_Lo-DNEBs6FSx-or3cfFxWXoO1kgYz0L9')

imgs =['http://i.imgur.com/xszf2ld.png',  # Stovetop with noodles
       'http://i.imgur.com/4ETgm9F.png',  # Salmon
       'http://i.imgur.com/mrVz0h0.png',  # Salmon Closeup
       'http://i.imgur.com/AqiBKzm.png',  # Noodles
       'http://i.imgur.com/erjfAaC.png',  # Boiled Egg
       'http://cdn1.medicalnewstoday.com/content/images/articles/245259-apples.jpg',  # Apple
       'http://cdn1.medicalnewstoday.com/content/images/articles/245259-broccoli.jpg',  # Brocolli
       ]

img_idx = 5

model = app.models.get('food-items-v1.0')
image = ClImage(url=imgs[img_idx])


predict = model.predict([image])

print('Prediction:')
for data in predict['outputs'][0]['data']['concepts'][0:5]:
    print('{} {}'.format(data['name'], data['value']))