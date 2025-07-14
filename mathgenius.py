# mathgenius.py

import streamlit as st
import sympy as sp
from sympy.parsing.sympy_parser import parse_expr
import numpy as np
import plotly.graph_objs as go
import time

# Streamlit page configs
st.set_page_config(
    page_title="MathGenius Pro",
    layout="wide",
    initial_sidebar_state="expanded",
    page_icon="🧮"
)

# CSS for dark theme and custom styling
dark_css = """
<style>
    :root {
      --primary-color: #7f5af0;
      --background-color: #121212;
      --card-bg: #1e1e2f;
      --text-color: #e0def4;
      --accent-color: #f6ad55;
      --error-color: #f56565;
      --border-radius: 12px;
    }

    .stApp {
      background-color: var(--background-color);
      color: var(--text-color);
      font-family: 'Inter', sans-serif;
      line-height: 1.5;
    }

    section[data-testid="stSidebar"] {
      background-color: var(--card-bg);
      border-radius: var(--border-radius);
      padding: 24px 16px 32px 24px;
    }

    .stButton > button {
      background-color: var(--primary-color);
      color: white;
      font-weight: 600;
      padding: 12px 24px;
      border-radius: var(--border-radius);
      border: none;
      transition: background-color 0.3s ease;
      width: 100%;
    }

    .stButton > button:hover {
      background-color: #6c4ee3;
    }

    .stTextInput > div > div > input {
      background-color: #27293d;
      color: var(--text-color);
      border-radius: var(--border-radius);
      border: 1px solid #44455a;
      padding: 12px 16px;
      font-size: 1rem;
    }

    .latex {
      background-color: #1e1e2f;
      border-radius: var(--border-radius);
      padding: 16px;
      font-size: 1.2rem;
      color: var(--accent-color);
    }

    .helper-text {
      font-size: 0.9rem;
      color: #c7c7c7;
      margin-top: -12px;
      margin-bottom: 20px;
      font-style: italic;
    }

    .stMarkdown h2 {
      border-bottom: 2px solid var(--primary-color);
      padding-bottom: 8px;
      margin-bottom: 16px;
      color: var(--primary-color);
    }
</style>
"""
st.markdown(dark_css, unsafe_allow_html=True)

# Helper functions
def parse_math_expression(expr_str, locals_dict={}):
    try:
        expr = parse_expr(expr_str, local_dict=locals_dict, evaluate=True)
        return expr
    except Exception as e:
        raise ValueError(f"Could not parse the expression:\n{e}")

def latex_display(expr):
    return f"$$\n{sp.latex(expr)}\n$$"

def solve_equation(input_text):
    try:
        systems_delimiters = [",", ";", "\n"]
        for delim in systems_delimiters:
            if delim in input_text:
                eqs_strings = [s.strip() for s in input_text.split(delim) if s.strip()]
                break
        else:
            eqs_strings = [input_text.strip()]

        variables = sorted(list({str(v) for eq in eqs_strings for v in sp.sympify(eq.replace('=', '-(')+')').free_symbols}))
        syms = sp.symbols(variables)

        eqs = []
        for eq_str in eqs_strings:
            if '=' in eq_str:
                left, right = eq_str.split('=')
                eq = parse_math_expression(f"({left.strip()}) - ({right.strip()})", {str(s): s for s in syms})
            else:
                eq = parse_math_expression(eq_str, {str(s): s for s in syms})
            eqs.append(eq)

        if len(eqs) == 1:
            eq = eqs[0]
            solutions = sp.solve(eq, syms if len(syms) > 1 else syms[0], dict=True)
        else:
            solutions = sp.solve(eqs, syms, dict=True)

        if not solutions:
            return "No solutions found."

        return solutions
    except Exception as e:
        raise ValueError(f"Could not solve the equation(s):\n{e}")

def differentiate_expression(expr_str):
    x = sp.symbols('x')
    expr = parse_math_expression(expr_str, local_dict={'x': x})
    deriv = sp.diff(expr, x)
    return deriv

def integrate_expression(expr_str, definite=False, lower=None, upper=None):
    x = sp.symbols('x')
    expr = parse_math_expression(expr_str, local_dict={'x': x})
    if definite:
        if lower is None or upper is None:
            raise ValueError("Definite integral requires lower and upper limits.")
        integral = sp.integrate(expr, (x, lower, upper))
    else:
        integral = sp.integrate(expr, x)
    return integral

def create_plot(expr_str):
    x = sp.symbols('x')
    expr = parse_math_expression(expr_str, local_dict={'x':x})
    func = sp.lambdify(x, expr, modules=["numpy"])

    x_vals = np.linspace(-10, 10, 500)
    try:
        y_vals = func(x_vals)
    except Exception as e:
        raise ValueError(f"Could not evaluate function for plotting:\n{e}")

    y_plot = np.array(y_vals, dtype=np.float64)
    mask = ~np.isnan(y_plot) & ~np.isinf(y_plot)
    x_vals = x_vals[mask]
    y_plot = y_plot[mask]

    fig = go.Figure()
    fig.add_trace(go.Scatter(x=x_vals, y=y_plot, mode='lines', line=dict(color="#7f5af0", width=3), name=str(expr)))
    fig.update_layout(
        template="plotly_dark",
        title="Function Plot",
        xaxis_title="x",
        yaxis_title="f(x)",
        margin=dict(l=40, r=40, t=60, b=40),
        height=400,
        hovermode="x unified"
    )
    return fig

def show_math_helper():
    st.markdown("""
        <div class='helper-text'>
        <strong>Math input tips:</strong><br>
        - Use **^** for powers, e.g. x^2<br>
        - Use `sin(x)`, `cos(x)`, `tan(x)`, `exp(x)`, `log(x)` for functions<br>
        - Use `=`, `/` for fractions<br>
        - Use `sqrt(x)` for square root<br>
        - Multiple equations can be separated by commas or semicolons for systems.<br>
        - For integrals, specify variable as 'x'. Definite integrals require limits.<br>
        </div>
        """, unsafe_allow_html=True)

# Main UI
def main():
    st.markdown("# MathGenius Pro - AI Math Companion")
    st.sidebar.title("Operations")
    operation = st.sidebar.selectbox(
        "Select an Operation",
        ["Solve Equations", "Differentiate Expression", "Integrate Expression", "Plot Function Graph"]
    )

    show_math_helper()
    expr_input = st.text_area("Enter your expression or equation:", height=100, max_chars=1000)

    if operation == "Integrate Expression":
        definite = st.checkbox("Definite Integral?")
        lower_limit = upper_limit = None
        if definite:
            col1, col2 = st.columns(2)
            with col1:
                try:
                    lower_limit = float(st.text_input("Lower limit (a):", "0"))
                except:
                    st.warning("Enter a valid number for lower limit")
            with col2:
                try:
                    upper_limit = float(st.text_input("Upper limit (b):", "1"))
                except:
                    st.warning("Enter a valid number for upper limit")

    if st.button("Compute"):
        try:
            if not expr_input.strip():
                st.warning("Please enter a valid mathematical expression.")
                return

            if operation == "Solve Equations":
                solutions = solve_equation(expr_input)
                if isinstance(solutions, str):
                    st.error(solutions)
                else:
                    st.success("Solution(s):")
                    for i, sol in enumerate(solutions):
                        for var, val in sol.items():
                            st.latex(f"{var} = {sp.latex(val)}")

            elif operation == "Differentiate Expression":
                derivative = differentiate_expression(expr_input)
                st.markdown("**Derivative:**")
                st.latex(sp.latex(derivative))

            elif operation == "Integrate Expression":
                integral = None
                if definite:
                    if lower_limit is None or upper_limit is None:
                        st.error("Please provide both lower and upper limits for definite integral.")
                        return
                    integral = integrate_expression(expr_input, definite=True, lower=lower_limit, upper=upper_limit)
                    st.markdown(f"**Definite Integral (from {lower_limit} to {upper_limit}):**")
                else:
                    integral = integrate_expression(expr_input, definite=False)
                    st.markdown("**Indefinite Integral:**")

                st.latex(sp.latex(integral))

            elif operation == "Plot Function Graph":
                fig = create_plot(expr_input)
                st.plotly_chart(fig, use_container_width=True)

        except Exception as e:
            st.error(f"Oops, an error occurred:\n{e}")

# Run the main function
if __name__ == "__main__":
    main()