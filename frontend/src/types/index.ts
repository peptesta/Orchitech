// --- TYPES ---
export interface ApiResponse {
  success: boolean;
  predicted_class: string;
  confidence: number;
  all_classes_probs: number[];
  image?: string; // Base64 of original image
  image_cropped?: string; // Base64 of the crop
  
  // Explainability fields (Original/Primary)
  integrated_gradients?: string; 
  occlusion?: string;

  // Comparison/Secondary fields (The ones causing the error)
  predicted_class_cropped?: string;
  confidence_cropped?: number;
  all_classes_probs_cropped?: number[];
  integrated_gradients_cropped?: string;
  occlusion_cropped?: string;

  // Error handling
  error?: string;
  traceback?: string;
}

export interface ImageFile {
  name: string;
  url: string;      
  file?: File;      
  analysis?: {
    boxes: number[][];    // Matrix Nx4
    scores: number[];     // List of N scores
    labels?: string[];    // Optional labels if your backend sends them
    count: number;        // Number N of detections
    
    // Tracking state
    modified?: boolean[];   // Tracks if a box position was edited
    eliminated?: boolean[]; // Tracks if a box was deleted/hidden
    isManual?: boolean[];   // Tracks if a box was manually drawn by the user
  };
}

export interface BackendResponse {
  images: string[];   
  bounding_box: number[][][]; 
  scores: number[][];         
  bb_count: number;           
}