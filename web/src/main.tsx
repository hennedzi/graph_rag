import { createRoot } from "react-dom/client";
import "@fontsource-variable/geist";
import "@fontsource-variable/geist-mono";
import App from "./App";
import "./index.css";

if (!window.location.hash) window.location.hash = "#/";
// светлая тема по умолчанию; выбор пользователя помним
document.documentElement.classList.toggle("dark", localStorage.getItem("theme") === "dark");

createRoot(document.getElementById("root")!).render(<App />);
