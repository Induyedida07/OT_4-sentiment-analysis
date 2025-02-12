from flask import Flask, render_template, request
from test import TextToNum
import pickle

app = Flask(__name__)

@app.route("/")
def Home():
    return render_template("index.html")

@app.route("/predict", methods=["GET", "POST"])
def predict():
    if request.method == "POST":
        msg = request.form.get("message")
        print("Message:", msg)

        # Process the input message
        cl = TextToNum(msg)
        cl.cleaner()
        cl.token()
        cl.removeStop()
        st = cl.stemme()
        stvc = " ".join(st)

        # Load Vectorizer and Model
        with open("vectorizer.pickle", "rb") as vc_file:
            vectorizer = pickle.load(vc_file)
        dt = vectorizer.transform([stvc]).toarray()

        with open("model.pickle", "rb") as mb_file:
            model = pickle.load(mb_file)
        
        # Get Prediction
        pred = model.predict(dt)
        prediction = str(pred[0])
        print("Prediction:", prediction)

        # Determine sentiment for display
        if prediction == "1":
            sentiment_text = "Positive Sentiment"
            sentiment_class = "positive"
            emoji = "😊"
        elif prediction == "0":
            sentiment_text = "Negative Sentiment"
            sentiment_class = "negative"
            emoji = "😞"
        else:
            sentiment_text = "Neutral Sentiment"
            sentiment_class = "neutral"
            emoji = "😐"
        
        # Pass the result to result.html
        return render_template("result.html", 
                               sentiment_text=sentiment_text, 
                               sentiment_class=sentiment_class, 
                               emoji=emoji)
    else:
        return render_template("predict.html")

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000)
