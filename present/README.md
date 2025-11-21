# Decentralized Federated Learning - Web Presentation

Modern, responsive HTML presentation for the Decentralized Federated Learning research project.

## 🎨 Features

- **Modern Dashboard Design**: Clean, flat design with pastel colors and rounded cards
- **10 Comprehensive Slides**: Covering all aspects of the research
- **Fully Responsive**: Works on desktop, tablet, and mobile devices
- **Multiple Navigation Methods**:
  - Navigation buttons (bottom center)
  - Keyboard shortcuts
  - Touch gestures (swipe)
  - Mouse wheel
- **Progress Tracking**: Visual progress bar at the top
- **Smooth Animations**: Professional transitions between slides

## 📋 Slide Contents

1. **Title Slide** - Project overview and team information
2. **Problem Statement** - IoT challenges and motivation
3. **DFL Solution** - Decentralized Federated Learning benefits
4. **Methodology** - Dataset and model architecture
5. **Experiments** - IID vs Non-IID setup
6. **Results** - Performance metrics and analysis
7. **Anomaly Detection** - Detection results (100% accuracy)
8. **Applications** - Real-world use cases
9. **Future Directions** - Research roadmap
10. **Conclusion** - Key achievements and contributions

## 🎮 Navigation Controls

### Keyboard Shortcuts
- **Arrow Keys** / **Space** / **Page Down**: Next slide
- **Arrow Up** / **Page Up**: Previous slide
- **Home**: Jump to first slide
- **End**: Jump to last slide
- **F**: Toggle fullscreen mode
- **?** or **H**: Show help dialog

### Mouse/Touch
- Click the **◀ ▶** navigation buttons at the bottom
- **Swipe left/right** on touch devices
- **Scroll** with mouse wheel (with delay to prevent accidental navigation)

## 🚀 How to Use

### Local Viewing
1. Open `index.html` in any modern web browser
2. Use navigation controls to move between slides
3. Press **F** for fullscreen presentation mode

### Web Hosting
Upload all files to any web server:
```bash
present/
├── index.html
├── style.css
└── script.js
```

### Presentation Tips
- Press **F** to enter fullscreen mode before presenting
- Use **Arrow Keys** or **Space** for smooth slide transitions
- The progress bar at the top shows your position in the presentation
- All content is fully responsive - works great on projectors and screens

## 🎨 Design Features

### Color Palette (Pastel)
- **Primary**: Purple gradient (#6C63FF)
- **Secondary**: Teal (#4ECDC4)
- **Success**: Mint green (#95E1D3)
- **Warning**: Yellow (#FFE66D)
- **Error**: Coral (#FF6B6B)

### UI Elements
- **Rounded Cards**: Soft corners for modern look
- **Pill-shaped Badges**: Category indicators
- **Icon Badges**: Colorful icons for visual hierarchy
- **Action Buttons**: Large, accessible call-to-action buttons
- **Two-column Layout**: Organized information display

## 📱 Responsive Breakpoints

- **Desktop**: 1200px+ (full layout)
- **Tablet**: 768px - 1199px (adjusted spacing)
- **Mobile**: < 768px (single column, optimized touch targets)

## 🌐 Browser Compatibility

Tested and working on:
- ✅ Chrome/Edge (latest)
- ✅ Firefox (latest)
- ✅ Safari (latest)
- ✅ Mobile browsers (iOS Safari, Chrome Mobile)

## 📄 File Structure

```
present/
├── index.html          # Main presentation file with all 10 slides
├── style.css           # Comprehensive styling with pastel theme
├── script.js           # Navigation logic and interactions
└── README.md          # This file
```

## ⌨️ Advanced Features

### Fullscreen Mode
Press **F** to toggle fullscreen for professional presentation.

### Time Tracking
The script automatically tracks time spent on each slide (visible in browser console).

### Print Support
Use **Ctrl+P** (Cmd+P on Mac) to print all slides for handouts.

### URL Navigation
Each slide has a unique URL hash (e.g., `#slide3`) for direct linking and browser history.

## 🎯 Key Highlights

### Slide 1: Title
- Team information with colored icon badges
- Course details in organized cards
- Large action buttons for navigation

### Slide 6: Results
- **18.52% improvement** highlighted
- Color-coded performance metrics
- Success indicators with green highlights

### Slide 7: Anomaly Detection
- **100% accuracy** emphasized
- Threshold visualization
- Color-coded anomaly types (red/orange/yellow)

### Slide 10: Conclusion
- Achievement badges
- Green success cards for key results
- Call-to-action buttons

## 🔧 Customization

### Changing Colors
Edit CSS variables in `style.css`:
```css
:root {
    --color-primary: #6C63FF;
    --color-secondary: #4ECDC4;
    /* ... more colors ... */
}
```

### Adding Slides
1. Copy a slide `<div class="slide">` in `index.html`
2. Update `totalSlides` in `script.js`
3. Add content following the existing card structure

### Modifying Content
Each slide uses a consistent structure:
- `.main-title` - Main heading
- `.sub-title` - Subtitle
- `.badges-row` - Category badges
- `.two-column-layout` - Content columns
- `.info-card` - Information cards with icons

## 📊 Performance

- **Load Time**: < 100ms on modern browsers
- **Animation**: Smooth 60fps transitions
- **File Size**: Minimal (< 100KB total)
- **No Dependencies**: Pure HTML/CSS/JS

## 🎓 Credits

Created for the Decentralized Federated Learning research project by Team 6.

**Team Members**: Trí Đông, Thành Phạm, Thu Thủy, Nguyễn Tâm, Justin  
**Advisor**: Trọng Nhân  
**Program**: Master of Data Science  
**Institution**: Bách Khoa University

---

**For questions or issues, please contact the development team.**

**Version**: 1.0  
**Last Updated**: November 2024
