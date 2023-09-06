from flask import Flask, request, render_template, redirect, url_for
from diffusers import StableDiffusionXLPipeline  # Assuming this is the right import
import torch
import openai
app = Flask(__name__)

@app.route('/')
def index():
    return render_template('form.html')

@app.route('/images')
def images():
    return render_template('images.html')

@app.route('/generate', methods=['POST'])
def generate_avatar():
    name = request.form['name']
    description = request.form['description']
    print(name, description)
    response = openai.Completion.create(
      engine="text-davinci-002",
      prompt=f"acting as a caricature artist describe what you would see as the looks of a sterotypical person named {name} who also embodies the ideas of \"{description}\". the description should be only a few sentences and not include the words \"caricature\", \"stereotypical\",or \"stereotype\" ",
      max_tokens=50
    )
    gpt3_prompt = response.choices[0].text.strip()
    # Prepare your pipeline
    base = StableDiffusionXLPipeline.from_single_file(
        './sd_xl_base_1.0_0.9vae.safetensors', 
        torch_dtype=torch.float16,
        variant="fp16",
        use_safetensors=True
    )
    base.to('cuda')
    
    # Use the user-provided description as the prompt
    # base_prompt = 'avatar ' + description
    # second_prompt = 'avatar ' + description + ', animated, happy, young, anime'
    # third_prompt = 'avatar ' + description + ', ugly, funny, farty, ascii art'
    
    # prompts = [base_prompt, second_prompt, third_prompt]
    neg_prompt = "low res, ugly, bad hands, too many digits, bad teeth, blurry, blurred background"
    image = base(prompt=gpt3_prompt, negative_prompt=neg_prompt).images[0]
    image.save('./static/avatar.png')

    # for i, prompt in enumerate(prompts):
        # image = base(prompt=prompt, negative_prompt=neg_prompt).images[0]
        # Assuming the image object has a save method, otherwise convert it to PIL image and save
        # image.save(f'{i}_img.png')
    return redirect(url_for("images"))
    # return f'Avatars generated! <a href="/static/avatar.png">View first image</a>'
    # return f'Avatar generated for {name} with the description: {description}'
@app.route('/static/<path:filename>')
def serve_static(filename):
    return send_from_directory('static', filename)
if __name__ == '__main__':
    app.run()
