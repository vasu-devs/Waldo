/** @type {import('tailwindcss').Config} */
export default {
    content: [
        "./index.html",
        "./src/**/*.{js,ts,jsx,tsx}",
    ],
    theme: {
        extend: {
            colors: {
                brand: {
                    purple: '#6750A4',
                    yellow: '#FBBF24',
                    pink: '#FF4D6D',
                    navy: '#0F0F1B',
                    black: '#1A1A1A',
                }
            },
            fontFamily: {
                lexend: ['Lexend', 'sans-serif'],
            },
            boxShadow: {
                'sos': '4px 4px 0px 0px #FF4D6D',
                'pill': '0 10px 15px -3px rgba(0, 0, 0, 0.1)',
            },
            borderRadius: {
                'pill': '9999px',
            }
        },
    },
    plugins: [],
}
