import { StrictMode } from "react";
import { createRoot } from "react-dom/client";
import "@fontsource/ibm-plex-sans/latin-400.css";
import "@fontsource/ibm-plex-sans/latin-500.css";
import "@fontsource/ibm-plex-sans/latin-600.css";
import "@fontsource/ibm-plex-mono/latin-400.css";
import "@fontsource/ibm-plex-mono/latin-500.css";
import App from "./App";
import "./styles/tokens.css";
import "./styles/base.css";
import "./styles/shell.css";
import "./styles/components.css";
import "./styles/charts.css";
import "./styles/pages.css";

createRoot(document.getElementById("root")!).render(
  <StrictMode>
    <App />
  </StrictMode>,
);
