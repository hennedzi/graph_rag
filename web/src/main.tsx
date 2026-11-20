import { createRoot } from "react-dom/client";
import "@fontsource-variable/geist";
import "@fontsource-variable/geist-mono";
import App from "./App";
import "./index.css";

if (!window.location.hash) window.location.hash = "#/";
// тёмная тема по умолчанию, как в Kilo; светлую пользователь включает явно
document.documentElement.classList.toggle("dark", localStorage.getItem("theme") !== "light");

createRoot(document.getElementById("root")!).render(<App />);
