import tkinter as tk
from tkinter import filedialog, messagebox
from PIL import Image, ImageTk, ImageDraw
import cv2
import numpy as np
from enhancement import apply_enhancement
from metrics import Metrics
from ultralytics import YOLO

def pil_to_cv2(pil_image):
    img_array = np.array(pil_image)
    if len(img_array.shape) == 2:  # Grayscale
        return cv2.cvtColor(img_array, cv2.COLOR_GRAY2BGR)
    elif img_array.shape[2] == 3:  # RGB
        return cv2.cvtColor(img_array, cv2.COLOR_RGB2BGR)
    elif img_array.shape[2] == 4:  # RGBA
        return cv2.cvtColor(img_array, cv2.COLOR_RGBA2BGR)
    return img_array


def cv2_to_pil(cv_image):
    rgb_image = cv2.cvtColor(cv_image, cv2.COLOR_BGR2RGB)
    return Image.fromarray(rgb_image)


# --- Main Application Controller ---
class ImageApp(tk.Tk):
    def __init__(self, *args, **kwargs):
        tk.Tk.__init__(self, *args, **kwargs)
        
        self.title("Image Enhancement with Object Detection Prototype")
        self.geometry("900x650")  
        
        container = tk.Frame(self)
        container.pack(side="top", fill="both", expand=True)
        container.grid_rowconfigure(0, weight=1)
        container.grid_columnconfigure(0, weight=1)

        self.frames = {}
        
        for F in (StartPage, EnhancementPage, ObjectDetectionPage):
            page_name = F.__name__
            frame = F(parent=container, controller=self)
            self.frames[page_name] = frame
            frame.grid(row=0, column=0, sticky="nsew")

        self.show_frame("StartPage")
        
    def show_frame(self, page_name, **kwargs):
        frame = self.frames[page_name]
        
        if page_name == "EnhancementPage" and "original_pil_image" in kwargs:
            frame.prepare_page(kwargs["original_pil_image"])
        elif page_name == "ObjectDetectionPage" and "enhanced_pil_image" in kwargs:
            frame.prepare_page(kwargs["enhanced_pil_image"])
            
        frame.tkraise()


# --- Start/Upload Page ---
class StartPage(tk.Frame):
    def __init__(self, parent, controller):
        tk.Frame.__init__(self, parent)
        self.controller = controller
        self.configure(bg="#b0bec5") 
        
        self.uploaded_image_pil = None
        self.uploaded_image_tk = None
        self.image_path = None
        
        self.main_frame = tk.Frame(self, bg="#b0bec5")
        self.main_frame.pack(fill=tk.BOTH, expand=True)
        
        self.upload_frame = tk.Frame(
            self.main_frame, 
            bg="white", 
            relief=tk.SUNKEN, 
            bd=5, 
            width=400,  
            height=300  
        )
        self.upload_frame.place(relx=0.5, rely=0.38, anchor=tk.CENTER)
        self.upload_frame.pack_propagate(False) 
        
        self.upload_area = tk.Label(
            self.upload_frame,
            text="Upload Image",
            font=("Arial", 28, "bold"), 
            bg="white",
            fg="black",
            cursor="hand2"
        )
        self.upload_area.pack(fill=tk.BOTH, expand=True) 
        self.upload_area.bind("<Button-1>", self.upload_image)
        
        self.button_frame = tk.Frame(self.main_frame, bg="#b0bec5")
        self.button_frame.place(relx=0.5, rely=0.82, anchor=tk.CENTER)
        
        self.reset_btn = tk.Button(
            self.button_frame,
            text="Reset",
            font=("Arial", 14),
            bg="#e57373", 
            fg="black",
            width=15,
            height=1,
            relief=tk.RIDGE, 
            bd=4, 
            cursor="hand2",
            command=self.reset_image_view
        )
        self.reset_btn.pack(side=tk.LEFT, padx=30, pady=10)
        
        self.enhance_btn = tk.Button(
            self.button_frame,
            text="Enhance Image",
            font=("Arial", 14),
            bg="#80deea", 
            fg="black",
            width=15,
            height=1,
            relief=tk.RIDGE, 
            bd=4, 
            cursor="hand2",
            command=self.enhance_image_and_switch
        )
        self.enhance_btn.pack(side=tk.LEFT, padx=30, pady=10)
        
    def upload_image(self, event=None):
        file_path = filedialog.askopenfilename(
            title="Select an Image",
            filetypes=[("Image Files", "*.png *.jpg *.jpeg *.bmp *.gif"), ("All Files", "*.*")]
        )
        
        if file_path:
            self.image_path = file_path
            try:
                self.uploaded_image_pil = Image.open(file_path)
                
                img_display = self.uploaded_image_pil.copy()
                img_display.thumbnail((380, 280), Image.Resampling.LANCZOS)
                
                self.uploaded_image_tk = ImageTk.PhotoImage(img_display)
                self.upload_area.configure(image=self.uploaded_image_tk, text="")
                self.upload_area.image = self.uploaded_image_tk
                
            except Exception as e:
                messagebox.showerror("Error", f"Could not load image: {str(e)}")
    
    def reset_image_view(self):
        self.upload_area.configure(image="", text="Upload Image")
        self.uploaded_image_pil = None
        self.uploaded_image_tk = None
        self.image_path = None
    
    def enhance_image_and_switch(self):
        if self.uploaded_image_pil:
            self.controller.show_frame("EnhancementPage", original_pil_image=self.uploaded_image_pil)
        else:
            messagebox.showwarning("Warning", "Please upload an image first!")


# --- Enhancement Metrics Page ---
class EnhancementPage(tk.Frame):
    def __init__(self, parent, controller):
        tk.Frame.__init__(self, parent)
        self.controller = controller
        self.configure(bg="#b0bec5") 
        
        self.original_pil_image = None
        self.enhanced_pil_image = None
        self.original_tk_image = None
        self.enhanced_tk_image = None
        
        # Initialize metrics calculator
        self.metrics_calculator = Metrics()
        
        # --- Original Image Section ---
        tk.Label(self, text="Original Image", font=("Arial", 16, "bold"), bg="#b0bec5", fg="black").place(relx=0.25, rely=0.08, anchor=tk.CENTER)
        
        self.original_frame = tk.Frame(self, bg="white", relief=tk.SUNKEN, bd=2, width=400, height=300)
        self.original_frame.place(relx=0.25, rely=0.35, anchor=tk.CENTER)
        self.original_frame.pack_propagate(False)
        
        self.original_label = tk.Label(self.original_frame, bg="white")
        self.original_label.pack(fill=tk.BOTH, expand=True)
        
        # --- Enhanced Image Section ---
        tk.Label(self, text="Enhanced Image", font=("Arial", 16, "bold"), bg="#b0bec5", fg="black").place(relx=0.75, rely=0.08, anchor=tk.CENTER)
        
        self.enhanced_frame = tk.Frame(self, bg="white", relief=tk.SUNKEN, bd=2, width=400, height=300)
        self.enhanced_frame.place(relx=0.75, rely=0.35, anchor=tk.CENTER)
        self.enhanced_frame.pack_propagate(False)
        
        self.enhanced_label = tk.Label(self.enhanced_frame, bg="white")
        self.enhanced_label.pack(fill=tk.BOTH, expand=True)
        
        # --- Main Title ---
        tk.Label(self, text="Image Enhancement Metrics Result:", font=("Arial", 18, "bold"), bg="#b0bec5", fg="black").place(relx=0.5, rely=0.64, anchor=tk.CENTER)
        
        # --- Metrics Labels ---
        self.entropy_orig = tk.Label(self, text="Original Entropy:", font=("Arial", 15), bg="#b0bec5", fg="black", anchor="w")
        self.entropy_orig.place(relx=0.08, rely=0.72, anchor=tk.W)
        
        self.entropy_enh = tk.Label(self, text="Enhanced Entropy:", font=("Arial", 15), bg="#b0bec5", fg="black", anchor="w")
        self.entropy_enh.place(relx=0.065, rely=0.78, anchor=tk.W)

        self.cii_label = tk.Label(self, text="Contrast Improvement Index:", font=("Arial", 15), bg="#b0bec5", fg="black", anchor="w")
        self.cii_label.place(relx=0.01, rely=0.84, anchor=tk.W)
        
        # --- Buttons ----
        tk.Button(
            self, text="Reset", font=("Arial", 14), bg="#e57373", fg="black", width=18, height=1,
            relief=tk.RIDGE, bd=3, cursor="hand2", command=self.reset_page 
        ).place(relx=0.77, rely=0.73, anchor=tk.CENTER)
        
        tk.Button(
            self, text="Object Detection", font=("Arial", 14), bg="#80deea", fg="black", width=18, height=1,
            relief=tk.RIDGE, bd=3, cursor="hand2", command=self.go_to_object_detection
        ).place(relx=0.77, rely=0.84, anchor=tk.CENTER)

    def go_to_object_detection(self):
        if self.enhanced_pil_image:
            self.controller.show_frame("ObjectDetectionPage", enhanced_pil_image=self.enhanced_pil_image)
        else:
            messagebox.showwarning("Warning", "No enhanced image available!")

    def reset_page(self):
        self.original_pil_image = None
        self.enhanced_pil_image = None
        self.original_tk_image = None
        self.enhanced_tk_image = None
        self.original_label.configure(image="")
        self.enhanced_label.configure(image="")
        self.entropy_orig.configure(text="Original Entropy:")
        self.entropy_enh.configure(text="Enhanced Entropy:")
        self.cii_label.configure(text="Contrast Improvement Index:")
        
        # Clear uploaded image in StartPage
        start_page = self.controller.frames["StartPage"]
        start_page.reset_image_view()
        
        # Go back to StartPage
        self.controller.show_frame("StartPage")
        
    def prepare_page(self, original_pil_image):
        self.original_pil_image = original_pil_image
        
        try:
            # Convert PIL to OpenCV format
            cv_image = pil_to_cv2(self.original_pil_image)
            
            # Convert to grayscale for metrics calculation
            gray_original = cv2.cvtColor(cv_image, cv2.COLOR_BGR2GRAY)
            
            # Apply enhancement using imported function
            enhanced_cv_image = apply_enhancement(cv_image)
            
            # Convert enhanced to grayscale for metrics
            gray_enhanced = cv2.cvtColor(enhanced_cv_image, cv2.COLOR_BGR2GRAY)
            
            # Convert back to PIL for display
            self.enhanced_pil_image = cv2_to_pil(enhanced_cv_image)
            
            # Calculate metrics
            entropy_original = self.metrics_calculator.calculate_entropy(gray_original)
            entropy_enhanced = self.metrics_calculator.calculate_entropy(gray_enhanced)
            cii_value = self.metrics_calculator.calculate_cii(gray_original, gray_enhanced)
            
            # Update metric labels
            self.entropy_orig.configure(text=f"Original Entropy: {entropy_original:.4f}")
            self.entropy_enh.configure(text=f"Enhanced Entropy: {entropy_enhanced:.4f}")
            self.cii_label.configure(text=f"Contrast Improvement Index: {cii_value:.4f}")
            
            # Display Original Image
            img_orig = self.original_pil_image.copy()
            img_orig.thumbnail((380, 280), Image.Resampling.LANCZOS)
            self.original_tk_image = ImageTk.PhotoImage(img_orig)
            self.original_label.configure(image=self.original_tk_image)
            self.original_label.image = self.original_tk_image
            
            # Display Enhanced Image
            img_enh = self.enhanced_pil_image.copy()
            img_enh.thumbnail((380, 280), Image.Resampling.LANCZOS)
            self.enhanced_tk_image = ImageTk.PhotoImage(img_enh)
            self.enhanced_label.configure(image=self.enhanced_tk_image)
            self.enhanced_label.image = self.enhanced_tk_image
            
        except Exception as e:
            messagebox.showerror("Enhancement Error", f"Failed to enhance image: {str(e)}")


# --- Object Detection Page ---
class ObjectDetectionPage(tk.Frame):
    def __init__(self, parent, controller):
        tk.Frame.__init__(self, parent)
        self.controller = controller
        self.configure(bg="#b0bec5")
        
        self.undetected_pil_image = None
        self.detected_pil_image = None
        self.undetected_tk_image = None
        self.detected_tk_image = None
        
        # Load YOLO model
        try:
            self.model = YOLO('best.pt')  
        except Exception as e:
            messagebox.showerror("Model Error", f"Failed to load YOLO model: {str(e)}")
            self.model = None
        
        # --- Undetected Image Section ---
        tk.Label(self, text="Undetected Image", font=("Arial", 16, "bold"), bg="#b0bec5", fg="black").place(relx=0.25, rely=0.08, anchor=tk.CENTER)
        
        self.undetected_frame = tk.Frame(self, bg="white", relief=tk.SUNKEN, bd=2, width=400, height=300)
        self.undetected_frame.place(relx=0.25, rely=0.35, anchor=tk.CENTER)
        self.undetected_frame.pack_propagate(False)
        
        self.undetected_label = tk.Label(self.undetected_frame, bg="white")
        self.undetected_label.pack(fill=tk.BOTH, expand=True)
        
        # --- Detection Image Section ---
        tk.Label(self, text="Detection Image", font=("Arial", 16, "bold"), bg="#b0bec5", fg="black").place(relx=0.75, rely=0.08, anchor=tk.CENTER)
        
        self.detected_frame = tk.Frame(self, bg="white", relief=tk.SUNKEN, bd=2, width=400, height=300)
        self.detected_frame.place(relx=0.75, rely=0.35, anchor=tk.CENTER)
        self.detected_frame.pack_propagate(False)
        
        self.detected_label = tk.Label(self.detected_frame, bg="white")
        self.detected_label.pack(fill=tk.BOTH, expand=True)
        
        # --- Main Title ---
        tk.Label(self, text="Object Detection Result", font=("Arial", 18, "bold"), bg="#b0bec5", fg="black").place(relx=0.5, rely=0.64, anchor=tk.CENTER)
        
        # --- Detection Results ---
        self.total_label = tk.Label(self, text="Number of objects detected: 0", font=("Arial", 15, "bold"), bg="#b0bec5", fg="black", anchor="w")
        self.total_label.place(relx=0.05, rely=0.71, anchor=tk.W)
        
        self.bus_label = tk.Label(self, text="Bus: 0", font=("Arial", 15), bg="#b0bec5", fg="black", anchor="w")
        self.bus_label.place(relx=0.16, rely=0.77, anchor=tk.W)
        
        self.cars_label = tk.Label(self, text="Cars: 0", font=("Arial", 15), bg="#b0bec5", fg="black", anchor="w")
        self.cars_label.place(relx=0.155, rely=0.82, anchor=tk.W)
        
        self.motor_label = tk.Label(self, text="Motor: 0", font=("Arial", 15), bg="#b0bec5", fg="black", anchor="w")
        self.motor_label.place(relx=0.145, rely=0.87, anchor=tk.W)
        
        self.truck_label = tk.Label(self, text="Truck: 0", font=("Arial", 15), bg="#b0bec5", fg="black", anchor="w")
        self.truck_label.place(relx=0.15, rely=0.92, anchor=tk.W)
        
        # Reset Button
        tk.Button(
            self, text="Reset", font=("Arial", 14), bg="#e57373", fg="black", width=18, height=1,
            relief=tk.RIDGE, bd=3, cursor="hand2", command=self.reset_page
        ).place(relx=0.77, rely=0.80, anchor=tk.CENTER)
    
    def reset_page(self):
        self.undetected_pil_image = None
        self.detected_pil_image = None
        self.undetected_tk_image = None
        self.detected_tk_image = None
        self.undetected_label.configure(image="")
        self.detected_label.configure(image="")
        self.total_label.configure(text="Number of objects detected: 0")
        self.bus_label.configure(text="Bus: 0")
        self.cars_label.configure(text="Cars: 0")
        self.motor_label.configure(text="Motor: 0")
        self.truck_label.configure(text="Truck: 0")
        
        # Clear all previous pages
        start_page = self.controller.frames["StartPage"]
        start_page.reset_image_view()
        
        enhancement_page = self.controller.frames["EnhancementPage"]
        enhancement_page.original_pil_image = None
        enhancement_page.enhanced_pil_image = None
        
        # Go back to StartPage
        self.controller.show_frame("StartPage")
    
    def prepare_page(self, enhanced_pil_image):
        self.undetected_pil_image = enhanced_pil_image
        
        try:
            # Display Undetected Image
            img_undetected = self.undetected_pil_image.copy()
            img_undetected.thumbnail((380, 280), Image.Resampling.LANCZOS)
            self.undetected_tk_image = ImageTk.PhotoImage(img_undetected)
            self.undetected_label.configure(image=self.undetected_tk_image)
            self.undetected_label.image = self.undetected_tk_image
            
            # Perform object detection
            if self.model:
                # Convert PIL to OpenCV format for YOLO
                cv_image = pil_to_cv2(self.undetected_pil_image)
                
                # Run YOLO detection
                results = self.model(cv_image, conf=0.25)  # confidence threshold 0.25
                
                # Initialize counters
                object_counts = {'bus': 0, 'car': 0, 'motor': 0, 'truck': 0}
                
                # Create a copy of the image for drawing
                detected_cv_image = cv_image.copy()
                
                # Process detections
                for result in results:
                    boxes = result.boxes
                    for box in boxes:
                        # Get box coordinates
                        x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()
                        x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)
                        
                        # Get class name and confidence
                        cls = int(box.cls[0])
                        conf = float(box.conf[0])
                        class_name = result.names[cls].lower()
                        
                        # Count objects
                        if class_name in object_counts:
                            object_counts[class_name] += 1
                        
                        # Draw bounding box
                        cv2.rectangle(detected_cv_image, (x1, y1), (x2, y2), (0, 255, 0), 2)
                        
                        # Draw label with class name and confidence
                        label = f"{class_name}: {conf:.2f}"
                        (text_width, text_height), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
                        cv2.rectangle(detected_cv_image, (x1, y1 - text_height - 10), (x1 + text_width, y1), (0, 255, 0), -1)
                        cv2.putText(detected_cv_image, label, (x1, y1 - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1)
                
                # Convert detected image back to PIL
                self.detected_pil_image = cv2_to_pil(detected_cv_image)
                
                # Calculate total
                total_objects = sum(object_counts.values())
                
                # Update detection counts with proper formatting
                self.total_label.configure(text=f"Number of objects detected: {total_objects}")
                self.bus_label.configure(text=f"Bus: {object_counts['bus']}")
                self.cars_label.configure(text=f"Cars: {object_counts['car']}")
                self.motor_label.configure(text=f"Motor: {object_counts['motor']}")
                self.truck_label.configure(text=f"Truck: {object_counts['truck']}")
                
                # Display Detection Image
                img_detected = self.detected_pil_image.copy()
                img_detected.thumbnail((380, 280), Image.Resampling.LANCZOS)
                self.detected_tk_image = ImageTk.PhotoImage(img_detected)
                self.detected_label.configure(image=self.detected_tk_image)
                self.detected_label.image = self.detected_tk_image
                
                # Print to console for debugging
                print(f"Detection Results: Total={total_objects}, Bus={object_counts['bus']}, Cars={object_counts['car']}, Motor={object_counts['motor']}, Truck={object_counts['truck']}")
                
            else:
                messagebox.showwarning("Warning", "YOLO model not loaded!")
            
        except Exception as e:
            messagebox.showerror("Detection Error", f"Failed to process image: {str(e)}")
            import traceback
            traceback.print_exc()


# --- Main Execution ---
if __name__ == "__main__":
    app = ImageApp()
    app.mainloop()