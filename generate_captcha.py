import os
import random
import string
import numpy as np
from PIL import Image, ImageDraw, ImageFont, ImageFilter
import pandas as pd

#Configuration
OUTPUT_DIR  = r'D:\Projects\Captcha 2\Data\raw'
LABELS_CSV  = r'D:\Projects\Captcha 2\Data\labels.csv'
NUM_IMAGES  = 100000
CAPTCHA_LEN = 5
IMG_WIDTH   = 200
IMG_HEIGHT  = 60

# Remove ambiguous characters
CHARSET = string.digits + string.ascii_lowercase
CHARSET = CHARSET.replace('0','').replace('o','').replace('1','').replace('l','')
print(f'Charset ({len(CHARSET)} chars): {CHARSET}')

os.makedirs(OUTPUT_DIR, exist_ok=True)


#Try to load common fonts, fallback to default if not found.
def get_font(size):
    font_paths = [
        "C:/Windows/Fonts/arial.ttf",
        "C:/Windows/Fonts/cour.ttf",
        "C:/Windows/Fonts/verdana.ttf",
        "C:/Windows/Fonts/times.ttf",
    ]
    for path in font_paths:
        if os.path.exists(path):
            try:
                return ImageFont.truetype(path, size)
            except:
                continue
    return ImageFont.load_default()

#Add random pixel noise.
def add_noise(img, noise_level=30):    
    arr   = np.array(img, dtype=np.int16)
    noise = np.random.randint(-noise_level, noise_level, arr.shape)
    arr   = np.clip(arr + noise, 0, 255).astype(np.uint8)
    return Image.fromarray(arr)

#Add random lines across the image.
def add_lines(draw, width, height, num_lines=4):
    for _ in range(num_lines):
        x1 = random.randint(0, width)
        y1 = random.randint(0, height)
        x2 = random.randint(0, width)
        y2 = random.randint(0, height)
        color = random.randint(80, 180)
        draw.line([(x1, y1), (x2, y2)], fill=color, width=random.randint(1, 2))

#Add random noise dots.
def add_dots(draw, width, height, num_dots=60):
    for _ in range(num_dots):
        x = random.randint(0, width)
        y = random.randint(0, height)
        r = random.randint(1, 3)
        color = random.randint(50, 200)
        draw.ellipse([(x-r, y-r), (x+r, y+r)], fill=color)

#Generate a CAPTCHA image with evenly spaced characters and noise.
def generate_captcha_image(label, width, height):
    
    bg_color = random.randint(230, 255)
    img  = Image.new('L', (width, height), color=bg_color)
    draw = ImageDraw.Draw(img)

    add_dots(draw, width, height, num_dots=40)
    add_lines(draw, width, height, num_lines=3)

    font_size  = random.randint(32, 40)
    font       = get_font(font_size)
    char_width = width // len(label)
    padding    = char_width // 6

    for i, char in enumerate(label):
        x = i * char_width + padding + random.randint(-3, 3)
        y = random.randint(5, height - font_size - 5)
        color = random.randint(20, 120)
        angle = random.randint(-12, 12)

        char_img  = Image.new('L', (char_width, height), color=bg_color)
        char_draw = ImageDraw.Draw(char_img)
        char_draw.text((padding, y - 5), char, fill=color, font=font)
        char_img  = char_img.rotate(angle, resample=Image.BILINEAR, expand=False, fillcolor=bg_color)
        img.paste(char_img, (i * char_width, 0))
        
    add_dots(draw, width, height, num_dots=20)

    if random.random() > 0.5:
        img = img.filter(ImageFilter.GaussianBlur(radius=0.8))

    img = add_noise(img, noise_level=20)
    return img


#Generate and save images with labels
records = []
for i in range(NUM_IMAGES):
    label    = ''.join(random.choices(CHARSET, k=CAPTCHA_LEN))
    filename = f'img_{i:05d}.png'
    filepath = os.path.join(OUTPUT_DIR, filename)

    img = generate_captcha_image(label, IMG_WIDTH, IMG_HEIGHT)
    img.save(filepath)

    records.append({'filename': filename, 'label': label})

    if (i + 1) % 1000 == 0:
        print(f'Generated {i+1}/{NUM_IMAGES}...')

df = pd.DataFrame(records)
df.to_csv(LABELS_CSV, index=False)
print(f'\nDone! {NUM_IMAGES} images saved.')
print(df.head(10))
