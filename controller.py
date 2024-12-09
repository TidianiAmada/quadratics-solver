from flask import Flask, render_template, request, jsonify
from newton_solver import newton_method_with_hessian

app = Flask(__name__)

@app.route("/", methods=["GET", "POST"])
def index():
    if request.method == "POST":
        try:
            # Parse form data
            function = request.form.get("function")
            variables = request.form.getlist("variables[]")  # Multiple variable names
            initial_guesses = list(map(float, request.form.getlist("initial_guesses[]")))
            tolerance = float(request.form.get("tolerance"))
            max_iterations = int(request.form.get("max_iterations"))

            # Solve using Newton's method
            result = newton_method_with_hessian(
                func_str=[function],  # Function should be a list for consistency
                variables=variables,
                initial_guess=initial_guesses,
                tolerance=tolerance,
                max_iterations=max_iterations,
            )

            return jsonify(result)
        except Exception as e:
            return jsonify({"status": "error", "message": str(e)})

    return render_template("index.html")


if __name__ == "__main__":
    app.run(debug=True)
