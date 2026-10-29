import { createRoot } from "react-dom/client";
import "@fontsource-variable/geist";
import "@fontsource-variable/geist-mono";
import App from "./App";
import "./index.css";

if (!window.location.hash) window.location.hash = "#/";
document.documentElement.classList.add("dark");

createRoot(document.getElementById("root")!).render(<App />);
