from flask import Flask, render_template, request
app = Flask(__name__)
import pickle

# open a file, where you stored the pickled data
file = open('model.pkl', 'rb')
# dump information to that file
clf = pickle.load(file)

# close the file
file.close()


@app.route('/', methods=["GET", "POST"])
def hello_world():
    if (request.method == "POST"):
        mydict=request.form
        fever=int(mydict['fever'])
        age=int(mydict['age'])
        pain=int(mydict['pain'])
        runnynose=int(mydict['runnynose'])
        diffbreath=int(mydict['diffbreath'])
        # code for inference
        inputfeatures=[fever, pain, age, runnynose, diffbreath]
        corona=clf.predict_proba([inputfeatures])[0][1]
        print(corona)
    # render on the template
        return render_template('show.html', inf=round(corona*100))
    return render_template('index.html')

    # return 'Hello, World!'+str(corona)



if __name__ == "__main__":
    app.run(debug=True)