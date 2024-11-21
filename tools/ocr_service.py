import base64
import io
from PIL import Image
import ollama
from app.utils import logger
import os
from datetime import datetime
import fitz

class OCRService:
    def __init__(self, model: str = "x/llama3.2-vision:latest"):
        self.model = model
        self.ollama_host = os.getenv("OLLAMA_HOST")
        self.ollama_port = os.getenv("OLLAMA_PORT")
        
    def encode_image_to_base64(self, image_path: str, format: str = "PNG") -> str:
        try:
            with Image.open(image_path) as img:
                buffered = io.BytesIO()
                img.save(buffered, format=format)
                return base64.b64encode(buffered.getvalue()).decode('utf-8')
        except Exception as e:
            logger.error(f"Error encoding image: {str(e)}")
            raise

    def process_document(self, file_path: str) -> str:
        """Process document and extract text using OCR."""
        try:
            base64_image = self.encode_image_to_base64(file_path)
            
            response = ollama.chat(
                model=self.model,
                messages=[{
                    "role": "user",
                    "content": "Extract all text from this document and format it as markdown. Include headers, lists, and tables if present.",
                    "images": [base64_image]
                }]
            )
            
            extracted_text = response.get('message', {}).get('content', '').strip()
            return extracted_text
            
        except Exception as e:
            logger.error(f"Error in OCR processing: {str(e)}")
            raise

    def insert_fitted_text(self, page, coords, text: str):
        """Insert text into a rectangle with proper fitting and scaling."""
        try:
            # Extract coordinates
            x0, y0, x1, y1 = coords
            rect_width = x1 - x0
            rect_height = y1 - y0

            # Add small margins
            margin = 5
            x0 += margin
            y0 += margin
            x1 -= margin
            y1 -= margin

            # Clean up text
            text = text.strip()
            if not text:
                logger.warning("Empty text provided for insertion")
                return

            # Start with a reasonable font size
            font_size = 11
            font = fitz.Font("helv")
            
            # Split text into words
            words = text.split()
            lines = []
            current_line = []
            current_width = 0
            
            # Form lines that fit the width
            for word in words:
                word_width = font.text_length(word + " ", fontsize=font_size)
                if current_width + word_width <= rect_width:
                    current_line.append(word)
                    current_width += word_width
                else:
                    if current_line:
                        lines.append(" ".join(current_line))
                    current_line = [word]
                    current_width = word_width
            
            if current_line:
                lines.append(" ".join(current_line))

            # Calculate total height needed
            line_height = font_size * 1.2
            total_height = len(lines) * line_height

            # Adjust font size if needed
            while total_height > rect_height and font_size > 6:
                font_size -= 0.5
                line_height = font_size * 1.2
                total_height = len(lines) * line_height

            logger.debug(f"Using font size {font_size} for {len(lines)} lines")

            # Calculate starting Y position to center text vertically
            y_start = y0 + (rect_height - total_height) / 2

            # Insert each line
            for i, line in enumerate(lines):
                y_pos = y_start + (i * line_height)
                page.insert_text(
                    point=(x0, y_pos + font_size),  # Add font_size to y_pos for baseline
                    text=line,
                    fontname="helv",
                    fontsize=font_size,
                    color=(0, 0, 0)  # Black text
                )

            logger.debug(f"Inserted {len(lines)} lines of text")
            return True

        except Exception as e:
            logger.error(f"Error inserting fitted text: {str(e)}")
            raise

    def process_pdf_with_ocr(self, file_path: str, temp_images_path: str):
        logger.info(f"Processing PDF file: {file_path}")
        pdf_document = None

        
        try:
            pdf_document = fitz.open(file_path)
            modified = False  # Track if we made any changes
            
            # Process each page
            for page_index in range(len(pdf_document)):
                page = pdf_document[page_index]
                image_list = page.get_images(full=True)
                
                if not image_list:
                    continue
                    
                # Process images on this page
                for image_index, img in enumerate(image_list, start=1):
                    try:
                        xref = img[0]
                        # Extract and process image
                        base_image = pdf_document.extract_image(xref)
                        image_bytes = base_image["image"]
                        image_ext = base_image["ext"]
                        
                        # Save image temporarily
                        save_img_path = os.path.join(
                            temp_images_path,
                            f"page_{page_index}_image_{image_index}.{image_ext}"
                        )
                        with open(save_img_path, "wb") as f:
                            f.write(image_bytes)
                            
                        # Get image rectangle
                        img_rect = page.get_image_rects(xref)[0]
                        coords = (img_rect.x0, img_rect.y0, img_rect.x1, img_rect.y1)
                        
                        # Process with OCR
                        logger.info(f"Processing image {save_img_path} with ocr")
                        ocr_text = self.process_document(save_img_path)
                        
                        if ocr_text.strip():
                            # Remove original image
                            page.delete_image(img[0])
                            
                            # Create white background
                            page.draw_rect(
                                coords,
                                color=(1, 1, 1),
                                fill=(1, 1, 1),
                                width=0
                            )
                            
                            # Insert OCR text
                            self.insert_fitted_text(page, coords, ocr_text)
                            modified = True
                            logger.info(f"Processed image {image_index} on page {page_index + 1}")
                            
                    except Exception as e:
                        logger.error(f"Error processing image: {str(e)}")
                        continue
                        
                    finally:
                        # Clean up temporary image
                        if os.path.exists(save_img_path):
                            os.remove(save_img_path)
            
            # After processing all pages, save if modified
            if modified:
                logger.info("Saving modified PDF")
                try:
                    # Create new filename with timestamp
                    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                    temp_path = f"{file_path}_modified_{timestamp}.pdf"
                    # Save to new file
                    pdf_document.save(
                        temp_path,
                        garbage=4,
                        deflate=True,
                        clean=True,
                        encryption=fitz.PDF_ENCRYPT_NONE
                    )
                    logger.debug(f"Saved modified PDF to: {temp_path}")

                    if pdf_document:
                        pdf_document.close()
                    # Verify the new PDF
                    if os.path.exists(temp_path) and os.path.getsize(temp_path) > 0:
                        # Replace original with modified version
                        os.replace(temp_path, file_path)
                        logger.info("Successfully replaced original PDF with modified version")
                    else:
                        raise ValueError("Modified PDF file is invalid or empty")
                        
                except Exception as e:
                    logger.error(f"Failed to save modified PDF: {str(e)}")
                    raise
                        
            else:
                logger.info("No modifications were made to the PDF")
                
        except Exception as e:
            logger.error(f"Error processing PDF file: {str(e)}")
            raise
            
        finally:    
            if temp_path and os.path.exists(temp_path):
                try:
                    os.remove(temp_path)
                    logger.debug("Cleaned up temporary PDF file")
                except Exception as e:
                    logger.warning(f"Failed to clean up temporary PDF: {str(e)}")

    def process_docx_with_ocr(self, file_path: str, temp_images_path: str):
        pass