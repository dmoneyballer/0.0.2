from flask import Flask, request, render_template, redirect, url_for, send_from_directory
from diffusers import StableDiffusionXLPipeline  # Assuming this is the right import
import torch
import openai
from threading import Thread

app = Flask(__name__)

global base

@app.route('/')
def index():
    return render_template('form.html')

@app.route('/images')
def images():
    return render_template('images.html')

@app.route('/generate', methods=['POST'])
def generate_avatar():
    Thread(target=long_generation_task, kwargs={'name':request.form['name'],'description':request.form['description']}).start()
    # TODO delete avatar images from static dir
    return redirect(url_for("images"))

def long_generation_task(name, description):
    print(f"name={name}, description={description}")

    neg_prompt = "low res, ugly, bad hands, too many digits, bad teeth, blurry, blurred background"

    response = openai.ChatCompletion.create(
      model="gpt-4-0613",
      messages=[{"role":"user","content":f'acting as a caricature artist describe what you would see as the looks of a sterotypical person named "{name}" who also embodies the ideas of "{description}". the description should only be a couple sentences, should not be longer than 70 words, and should not include the words "caricature", "stereotypical",or "stereotype"'}],
      temperature = 0.7
    )
    gpt4_response = response.choices[0].message.content.strip()
    print(f"gpt4_response={gpt4_response}")

    image = base(prompt=f"avatar of {name}. {gpt4_response}", negative_prompt=neg_prompt).images[0]
    image.save('./static/avatar.png')

    # Use the user-provided description as the prompt
    prompts = [
        f"avatar {description}",
        f"avatar {description}, animated, happy, young, anime",
        f"avatar {description}, ugly, funny, farty, ascii art",
        f'profile avatar in the style of "Avatar: the Last Airbender". the image should be for a user named "{name}" and should embody the essence of "{description}"',
        f'3d, style of funko pop toy, caricature, named "{name}" and should embody the essence of "{description}" '
                ]

    for i, prompt in enumerate(prompts):
        print(f"prompt={prompt}")
        image = base(prompt=prompt, negative_prompt=neg_prompt).images[0]
        #Assuming the image object has a save method, otherwise convert it to PIL image and save
        image.save(f"./static/avatar{i}.png")

@app.route('/static/<path:filename>')
def serve_static(filename):
    return send_from_directory('static', filename)

if __name__ == '__main__':
    print("Preparing StableDiffusion pipeline…")
    base = StableDiffusionXLPipeline.from_single_file(
        './sd_xl_base_1.0_0.9vae.safetensors',
        torch_dtype=torch.float16,
        variant="fp16",
        use_safetensors=True,
    )
    base=base.to('cuda')

    print("run flask app")
    app.run()
