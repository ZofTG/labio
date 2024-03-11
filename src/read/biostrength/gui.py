"""gui module"""

#! imports


import tkinter as tk
from os import makedirs, sep
from os.path import join, dirname
from tkinter import PhotoImage, filedialog, messagebox, ttk

from .products import PRODUCTS
from .assets import ICON64


#! functions


def run():
    """generate the Bioreader GUI"""

    # main window
    window = tk.Tk()
    window.geometry("400x100")
    window.iconphoto(True, PhotoImage(data=ICON64))
    window.title("Bioreader")

    # input frame
    input_frame = ttk.LabelFrame(master=window, text="Input data")
    input_frame.pack(side="top", fill="both", expand=True)
    label_text = tk.StringVar(value="")

    def open_file():
        file = filedialog.askopenfilename(
            parent=window,
            initialdir=join(dirname(__file__)),
            title="Please select a file:",
            filetypes=[("text files", ".txt")],
        )
        label_text.set(file)

    button = ttk.Button(
        master=input_frame,
        text="Import file",
        command=open_file,
    )
    button.pack(side="left")
    label = ttk.Label(master=input_frame, textvariable=label_text)
    label.pack(side="right")

    # output frame
    output_frame = ttk.LabelFrame(master=window, text="Output data")
    output_frame.pack(side="bottom", fill="both", expand=True)
    pr_label = ttk.Label(master=output_frame, text="Product: ")
    pr_label.pack(side="left")
    products = list(PRODUCTS.keys())
    product = tk.StringVar(value=products[0])
    prods = [products[0]] + products
    dropdown = ttk.OptionMenu(output_frame, product, *prods)
    dropdown.pack(side="left")
    dropdown.config(padding=2)

    def save_file():
        try:
            # convert raw data into user-understandable data
            data = PRODUCTS[product.get()].from_file(label_text.get())

            # make the output file
            keys = list(data.keys())
            lines = [",".join([i.split("_")[0] for i in keys])]
            lines += [",".join([i.split("_")[1] for i in keys])]
            values = list(data.values())
            for j in range(len(values[0])):
                line = ",".join([str(values[i][j]) for i in range(len(keys))])
                lines += [line]
            txt = "\n".join(lines)

            # check where to save the data
            initial = label_text.get().rsplit(".", 1)[0] + "_converted.csv"
            file = filedialog.asksaveasfilename(
                parent=window,
                confirmoverwrite=True,
                filetypes=[("CSV comma-separated files", ".csv")],
                initialfile=initial,
                title="Please select the file to be saved:",
            )
            file = file.replace("/", sep)

            # save the converted data
            if len(file) > 0:
                if not file.endswith(".csv"):
                    file += ".csv"
                makedirs(file.rsplit(sep, 1)[0], exist_ok=True)
                with open(file, "wt", encoding="utf-8") as buf:
                    buf.write(txt)
                messagebox.showinfo("Info", "Conversion complete")
        except Exception as exc:
            messagebox.showerror(
                title="Error",
                message=exc.args[0],
            )

    save = ttk.Button(
        master=output_frame,
        text="Convert",
        command=save_file,
    )
    save.pack(side="right")

    window.mainloop()


#! MAIN

if __name__ == "__main__":
    run()
