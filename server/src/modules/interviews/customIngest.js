import multer from "multer";
import pdf from "pdf-parse";
import mammoth from "mammoth";
import AppError from "../../utils/AppError.js";
import asyncHandler from "../../utils/asyncHandler.js";
import { ingestCustomText } from "../../integrations/aiInterviewService.js";

// Keep files in memory for fast serverless parsing
const upload = multer({
  storage: multer.memoryStorage(),
  limits: { fileSize: 5 * 1024 * 1024 } // 5 MB Limit
});

export const parseCustomNotesUpload = upload.single("notes");

export const handleCustomNotesIngest = asyncHandler(async (req, res, next) => {
  if (!req.file) {
    return next(new AppError("No file uploaded. Please upload a PDF, DOCX, TXT, or MD file.", 400));
  }

  const { originalname, buffer, mimetype } = req.file;
  let text = "";

  try {
    if (mimetype === "application/pdf" || originalname.endsWith(".pdf")) {
      const parsedPdf = await pdf(buffer);
      text = parsedPdf.text || "";
    } else if (
      mimetype === "application/vnd.openxmlformats-officedocument.wordprocessingml.document" || 
      originalname.endsWith(".docx")
    ) {
      const parsedWord = await mammoth.extractRawText({ buffer });
      text = parsedWord.value || "";
    } else if (
      mimetype === "text/plain" || 
      originalname.endsWith(".txt") || 
      originalname.endsWith(".md")
    ) {
      text = buffer.toString("utf-8");
    } else {
      return next(new AppError("Unsupported file type. Only PDF, DOCX, TXT, and Markdown files are supported.", 400));
    }
  } catch (err) {
    return next(new AppError(`Failed to extract text from file: ${err.message}`, 400));
  }

  const cleanText = text.trim();
  if (cleanText.length < 50) {
    return next(new AppError("The uploaded file is too short or contains no readable text.", 400));
  }

  // Generate a unique topic ID for this study guide
  const topicId = `custom_notes_${req.user._id}_${Date.now()}`;

  try {
    // Send extracted text to Python vector engine
    await ingestCustomText(topicId, cleanText);

    return res.status(200).json({
      success: true,
      message: "Custom study guide ingested successfully",
      topic: topicId
    });
  } catch (err) {
    return next(new AppError(`Failed to ingest custom notes into AI engine: ${err.message}`, 500));
  }
});
