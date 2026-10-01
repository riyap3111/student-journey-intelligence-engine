import { Route, BrowserRouter as Router, Routes } from "react-router-dom";
import Navbar from "./components/Navbar";
import Home from "./pages/Home";
import ModelInfo from "./pages/ModelInfo";
import Monitoring from "./pages/Monitoring";
import Predict from "./pages/Predict";

export default function App() {
  return (
    <Router>
      <Navbar />
      <Routes>
        <Route path="/" element={<Home />} />
        <Route path="/predict" element={<Predict />} />
        <Route path="/model" element={<ModelInfo />} />
        <Route path="/monitoring" element={<Monitoring />} />
      </Routes>
    </Router>
  );
}
