/*
 * Licensed to the Apache Software Foundation (ASF) under one
 * or more contributor license agreements.  See the NOTICE file
 * distributed with this work for additional information
 * regarding copyright ownership.  The ASF licenses this file
 * to you under the Apache License, Version 2.0 (the
 * "License"); you may not use this file except in compliance
 * with the License.  You may obtain a copy of the License at
 *
 *   http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing,
 * software distributed under the License is distributed on an
 * "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY
 * KIND, either express or implied.  See the License for the
 * specific language governing permissions and limitations
 * under the License.
 */

(function () {
    var saved = localStorage.getItem("theme");
    if (saved) {
        document.documentElement.setAttribute("data-theme", saved);
    } else if (window.matchMedia("(prefers-color-scheme: dark)").matches) {
        document.documentElement.setAttribute("data-theme", "dark");
    }

    document.addEventListener("DOMContentLoaded", function () {
        var btn = document.querySelector(".theme-toggle");
        if (btn) {
            btn.addEventListener("click", function () {
                var current = document.documentElement.getAttribute("data-theme");
                var next = current === "dark" ? "light" : "dark";
                document.documentElement.setAttribute("data-theme", next);
                localStorage.setItem("theme", next);
                btn.textContent = next === "dark" ? "Light Mode" : "Dark Mode";
            });
            var theme = document.documentElement.getAttribute("data-theme");
            btn.textContent = theme === "dark" ? "Light Mode" : "Dark Mode";
        }

        var toggle = document.querySelector(".menu-toggle");
        var sidebar = document.querySelector(".sidebar");
        var overlay = document.querySelector(".overlay");
        if (toggle && sidebar) {
            toggle.addEventListener("click", function () {
                sidebar.classList.toggle("open");
                if (overlay) overlay.classList.toggle("open");
            });
            if (overlay) {
                overlay.addEventListener("click", function () {
                    sidebar.classList.remove("open");
                    overlay.classList.remove("open");
                });
            }
        }

        var observer = new IntersectionObserver(function (entries) {
            entries.forEach(function (entry) {
                if (entry.isIntersecting) {
                    entry.target.classList.add("visible");
                    observer.unobserve(entry.target);
                }
            });
        }, { threshold: 0.1, rootMargin: "0px 0px -40px 0px" });

        document.querySelectorAll(".feature, .arch-svg, table, .note, .warning, .tip").forEach(function (el) {
            el.classList.add("fade-in");
            observer.observe(el);
        });

        var style = document.createElement("style");
        style.textContent = ".fade-in{opacity:0;transform:translateY(12px);transition:opacity .45s cubic-bezier(.4,0,.2,1),transform .45s cubic-bezier(.4,0,.2,1)}.fade-in.visible{opacity:1;transform:translateY(0)}";
        document.head.appendChild(style);
    });
})();
