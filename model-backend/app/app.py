from flask import Flask, request, jsonify

app = Flask(__name__)

# In-memory database for simplicity
data = []

# Home route
@app.route('/')
def home():
    return 'Welcome To, Prostate xAi!'

# Create a book
@app.route('/model', methods=['POST'])
def model():
    # Get the form data
    title = request.form['title']
    username = request.form['username']
    # Get the image file
    scan = request.files['file_image']
    
    # Process the image file (e.g., save it to a directory)
    if scan:
        image_filename = f"{len(data) + 1}_{scan.filename}"
        scan.save(f"uploads/{image_filename}")
    else:
        image_filename = None

    # Create the book dictionary
    new_data = {
        'id': len(data) + 1,
        'title': title,
        'author': username,
        'scan': image_filename
    }

    # Append the book to the list
    data.append(new_data)

    # Return the created book
    return jsonify(new_data), 201

if __name__ == '__main__':
    app.run(debug=True)
