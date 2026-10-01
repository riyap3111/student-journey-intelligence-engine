import { Route, BrowserRouter as Router, Routes } from "react-router-dom";
import Footer from "./components/Footer";
import Navbar from "./components/Navbar";
import Home from "./pages/Home";
import ModelInfo from "./pages/ModelInfo";
import Monitoring from "./pages/Monitoring";
import Predict from "./pages/Predict";

export default function App() {
  return (
    <Router>
      <div className="flex min-h-screen flex-col">
        <Navbar />
        <div className="flex-1">
          <Routes>
            <Route path="/" element={<Home />} />
            <Route path="/predict" element={<Predict />} />
            <Route path="/model" element={<ModelInfo />} />
            <Route path="/monitoring" element={<Monitoring />} />
          </Routes>
        </div>
        <Footer />
      </div>
    </Router>
  );
}
