// --- TYPES ---
export interface ApiResponse {
  predicted_class: string;
  confidence: number;
  all_classes_probs: number[];
  image_cropped?: string;
  error?: string;
  
  // Compare Mode Data
  predicted_external_class?: string;
  confidence_external?: number;
  all_classes_probs_external?: number[];
  
  // Explainability
  integrated_gradients?: string;
  occlusion?: string;
  integrated_gradients_external?: string;
  occlusion_external?: string;
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